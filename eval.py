from train import *
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.nn import BCEWithLogitsLoss

class MetaLogReg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear = nn.Linear(num_classes * 2, num_classes)

    def forward(self, audio_probs, prior_probs):
        x = torch.cat([audio_probs, prior_probs], dim=1)
        return self.linear(x)

def get_priorvec(i):
    lon = test_df.iloc[i]["longitude"].round(0)
    lat = test_df.iloc[i]["latitude"].round(0)
    species = test_df.iloc[i]["primary_label"]

    priorvec = torch.full((NUM_CLASSES,), 1.0 / NUM_CLASSES)

    try:
        loc_prior = prior.loc[(lat, lon)]

        for species, p in loc_prior.items():
            priorvec[label2id[species]] = p

    except KeyError:
        pass
    return priorvec

def get_priorvec_batch(i, df, b=BATCH_SIZE):
    batch = df.iloc[i*b:i*b+b]

    prior_list = []

    for _, row in batch.iterrows():

        lat = round(row["latitude"])
        lon = round(row["longitude"])

        priorvec = torch.full((NUM_CLASSES,), 1.0 / NUM_CLASSES)

        try:
            loc_prior = prior.loc[(lat, lon)]

            for species, p in loc_prior.items():
                priorvec[label2id[species]] = p

        except KeyError:
            pass

        prior_list.append(priorvec)

    return torch.stack(prior_list)   # <-- THIS gives (B, C)

if __name__=="__main__":

    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    meta_model = MetaLogReg(NUM_CLASSES).to("cpu")
    meta_model.load_state_dict(
        torch.load("meta_model.pth")
    )
    test_persample_loss = 0.0
    total_samples = 0

    sol = pd.read_csv("../data/sample_submission.csv")
    print(sol.shape)
    print(df.select_dtypes(include="number").iloc[0].sum())

    DEVICE = "cpu"
    model = BirdModel(NUM_CLASSES)
    model = model.to(DEVICE)
    model.load_state_dict(
        torch.load("../bird_model5K20260529_002038.pth")
    )
    df, label2id = df_to_species(test_df)
    y_predlist_audio = []
    y_predlist_fixed = []
    y_predlist_meta = []
    y_truelist = []

    for i, (x, y) in tqdm(enumerate(test_loader)):
        x = x.to(DEVICE)
        model.eval()
        """ GET PRIOR PROBABILITY """
        if (i*BATCH_SIZE+BATCH_SIZE) > df.shape[0]:
            continue
        priorvec = get_priorvec_batch(i, df)

        with torch.no_grad():
            logits_audio = model(x)

        """ FORWARD PASS META MODEL """
        p_audio = torch.sigmoid(logits_audio)
        meta_logits = meta_model(p_audio, priorvec)

        """ FIXED MIX """
        alpha = 0.5
        logits_fixed = alpha * logits_audio + (1 - alpha) * priorvec

        # If you need the class id: 78
        y_pred_meta = torch.argmax(meta_logits, dim=1).cpu().numpy()
        y_pred_fixed = torch.argmax(logits_fixed, dim=1).cpu().numpy()
        y_pred_audio = torch.argmax(logits_audio, dim=1).cpu().numpy()
        y_true = torch.argmax(y, dim=1).cpu().numpy()



        y_predlist_audio.extend(y_pred_audio)
        y_predlist_fixed.extend(y_pred_fixed)
        y_predlist_meta.extend(y_pred_meta)
        y_truelist.extend(y_true)
        #print(f" Mixed prob: {torch.round(mixed_prob * 100) / 100} Audio prob: {torch.round(torch.tensor(probs) * 100) / 100} Prior value: {torch.round(torch.tensor(priorprob) * 100) / 100}")

    print(f"test per sample loss {test_persample_loss:.4f} | ")
    # 4. Calculate metrics. Compute binary probvector from logits (model output)
    #predictions = (np.array(logitslist) > 0.0).astype(int)

    precision = precision_score(y_truelist,y_predlist_audio,average="macro",zero_division=0)
    recall = recall_score(y_truelist,y_predlist_audio,average="macro",zero_division=0)
    f1 = f1_score(y_truelist,y_predlist_audio,average="macro",zero_division=0)
    print(precision, recall, f1)

    precision = precision_score(y_truelist,y_predlist_fixed,average="macro",zero_division=0)
    recall = recall_score(y_truelist,y_predlist_fixed,average="macro",zero_division=0)
    f1 = f1_score(y_truelist,y_predlist_fixed,average="macro",zero_division=0)
    print(precision, recall, f1)

    precision = precision_score(y_truelist,y_predlist_meta,average="macro",zero_division=0)
    recall = recall_score(y_truelist,y_predlist_meta,average="macro",zero_division=0)
    f1 = f1_score(y_truelist,y_predlist_meta,average="macro",zero_division=0)
    print(precision, recall, f1)

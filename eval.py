from train import *
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.nn import BCEWithLogitsLoss

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

def get_priorvec_batch(i, b=BATCH_SIZE):
    batch = train_df.iloc[i*b:i*b+b]

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
    y_pred = []
    y_true = []

    for i, (x, y) in tqdm(enumerate(test_loader)):
        x = x.to(DEVICE)
        model.eval()
        """ FORWARD PASS """

        with torch.no_grad():
            logits = model(x)

        p_audio = torch.sigmoid(logits).unsqueeze(0)

        #probs = torch.where(probs < 1e-1,torch.tensor(0.0, device=probs.device),probs)
        pred_species = np.argmax(p_audio)

        """ GET PRIOR PROBABILITY """
        priorvec = get_priorvec(i).unsqueeze(0)

        alpha = 0.5
        mixed_probs = p_audio * (priorvec.to(p_audio.device) ** alpha)

        #y_pred = torch.argmax(mixed_probs).item()
        #print(f" priorprob: {priorprob} y_pred: {int(probs[pred_species])} y: {np.argmax(y)}")
        pred = torch.argmax(mixed_probs).item()
        true = torch.argmax(y).item()

        y_pred.append(pred)
        y_true.append(true)
        #print(f" Mixed prob: {torch.round(mixed_prob * 100) / 100} Audio prob: {torch.round(torch.tensor(probs) * 100) / 100} Prior value: {torch.round(torch.tensor(priorprob) * 100) / 100}")

    print(f"test per sample loss {test_persample_loss:.4f} | ")
    # 4. Calculate metrics. Compute binary probvector from logits (model output)
    #predictions = (np.array(logitslist) > 0.0).astype(int)

    precision = precision_score(y_true,y_pred,average="macro",zero_division=0)
    recall = recall_score(y_true,y_pred,average="macro",zero_division=0)
    f1 = f1_score(y_true,y_pred,average="macro",zero_division=0)

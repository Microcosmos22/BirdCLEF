from train import *
from sklearn.metrics import precision_score, recall_score, f1_score

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
    logitslist = []
    ytruelist = []

    for i, (x, y) in tqdm(enumerate(test_loader)):
        x = x.to(DEVICE)

        model.eval()

        with torch.no_grad():
            logits = model(x)
            logitslist.append(logits)

        probs = torch.sigmoid(logits)
        probs = torch.where(
            probs < 1e-1,
            torch.tensor(0.0, device=probs.device),
            probs
        )
        y_pred = np.argmax(probs, axis=1)
        ytruelist.append(y)
        loss = criterion(logits, y)

        test_persample_loss += loss.item()/len(test_loader)



        lon = test_df.iloc[i]["longitude"].round(0)
        lat = test_df.iloc[i]["latitude"].round(0)
        species = test_df.iloc[i]["primary_label"]

        try:
            priorprob = prior.loc[(lat, lon, species)]
        except KeyError:
            priorprob = global_prior.get(species, 1.0 / NUM_CLASSES)

        id = label2id[species]
        priorvec = torch.zeros((NUM_CLASSES))
        priorvec[id] = 1

        print(id)
        print(priorvec)

        alpha = 0.5
        mixed_prob = probs * (priorprob ** alpha)

        #print(f" Mixed prob: {torch.round(mixed_prob * 100) / 100} Audio prob: {torch.round(torch.tensor(probs) * 100) / 100} Prior value: {torch.round(torch.tensor(priorprob) * 100) / 100}")


    print(
        f"test per sample loss {test_persample_loss:.4f} | ")
    # 4. Calculate metrics

    # 4. Calculate metrics. Compute binary probvector from logits (model output)
    predictions = (np.array(logitslist) > 0.0).astype(int)

    precision = precision_score(ytruelist, logitslist)
    recall = recall_score(ytruelist, logitslist)
    f1 = f1_score(ytruelist, logitslist)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    """ ######### """
    model.load_state_dict(
        torch.load("../bird_model5K20260528_233112.pth")
    )

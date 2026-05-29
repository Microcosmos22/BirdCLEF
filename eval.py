from train import *

if __name__=="__main__":

    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    val_persample_loss = 0.0
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


    for x, y in tqdm(val_loader):
        x = x.to(DEVICE)

        model.eval()

        with torch.no_grad():
            logits = model(x)

        probs = torch.sigmoid(logits)
        probs = torch.where(
            probs < 1e-1,
            torch.tensor(0.0, device=probs.device),
            probs
        )

        loss = criterion(logits, y)

        val_persample_loss += loss.item()/len(val_loader)

    print(
        f"Val per sample loss {val_persample_loss:.4f} | ")

    """ ######### """
    model.load_state_dict(
        torch.load("../bird_model5K20260528_233112.pth")
    )


    for x, y in tqdm(val_loader):
        x = x.to(DEVICE)

        model.eval()

        with torch.no_grad():
            logits = model(x)

        probs = torch.sigmoid(logits)
        probs = torch.where(
            probs < 1e-1,
            torch.tensor(0.0, device=probs.device),
            probs
        )

        loss = criterion(logits, y)

        val_persample_loss += loss.item()/len(val_loader)

    print(
        f"Val per sample loss {val_persample_loss:.4f} | ")

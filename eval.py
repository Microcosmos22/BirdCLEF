from train import *

if __name__=="__main__":

    sol = pd.read_csv("../data/sample_submission.csv")
    print(sol.shape)
    print(df.select_dtypes(include="number").iloc[0].sum())

    DEVICE = "cpu"
    model = BirdModel(NUM_CLASSES)
    model = model.to(DEVICE)
    model.load_state_dict(
        torch.load("bird_model20260526_195539.pth")
    )


    for x, y in tqdm(train_loader):
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
        print(probs)

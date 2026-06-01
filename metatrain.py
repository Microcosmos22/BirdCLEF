from train import *
from eval import *
from torch import nn
from torch.optim import Adam

if __name__ == "__main__":
    DEVICE = "cpu"

    meta_model = MetaLogReg(NUM_CLASSES).to(DEVICE)
    optimizer = Adam(meta_model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    """ Model for audio preds """
    model = BirdModel(NUM_CLASSES)
    model = model.to(DEVICE)
    model.load_state_dict(
        torch.load("../bird_model5K20260529_002038.pth")
    )

    meta_model.train()

    logitslist_train_pred_5K002038model = []

    for epoch in range(1):

        total_loss = 0

        for i, (x, y) in tqdm(enumerate(train_loader)):
            x = x.to(DEVICE)
            y = y.to(DEVICE).float()

            with torch.no_grad():
                logits = model(x)
                logitslist_train_pred_5K002038model.append(logits)

            audio_probs = torch.sigmoid(logits)
            prior_probs = get_priorvec_batch(i)

            preds = meta_model(audio_probs, prior_probs)

            loss = criterion(preds, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 30 == 0:
                np.save("modeloutputs.npy", np.asarray(logitslist_train_pred_5K002038model))
                torch.save(
                    meta_model.state_dict(),
                    "meta_model.pth"
                )

        print("epoch", epoch, "loss", total_loss)

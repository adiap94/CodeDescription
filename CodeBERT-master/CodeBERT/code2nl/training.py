import os
import torch
import pandas as pd
import numpy as np
class train_class():

    def __init__(self, model, optimizer, train_loader, val_loader,epoch_num, out_dir, device,scheduler):
        epoch = 0
        self.device = device
        self.epoch_log = {}
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epoch_num = epoch_num
        self.out_dir = out_dir
        self.csvLoggerFile_path = os.path.join(out_dir, "history.csv")
        self.best_metric = np.inf




    def writeCSVLoggerFile(self):
        df = pd.DataFrame([self.epoch_log])
        with open(self.csvLoggerFile_path, 'a') as f:
            df.to_csv(f, mode='a', header=f.tell() == 0, index=False)

    def save_model_checkpoint(self):
        PATH = os.path.join(self.out_dir, "Models", "last_model.pt")
        torch.save(self.model, PATH)
        print("saved last model")

        if self.epoch_log["val_loss"] < self.best_metric:
            self.best_metric = self.epoch_log["val_loss"]
            PATH = os.path.join(self.out_dir, "Models", "best_metric_model.pt")
            torch.save(self.model, PATH)
            print("saved new best metric model")


    def train(self):

        for epoch in range(self.epoch_num):
            self.epoch_log = {}
            self.epoch_log["epoch"] = epoch
            print(f"epoch {epoch}/{self.epoch_num}")

            self.model.train()

            train_loss = 0
            step = 0
            for batch_data in self.train_loader:
                step += 1

                # print(step)
                source_ids, source_mask,target_ids,target_mask = (
                    batch_data["source_ids"].to(self.device),
                    batch_data["source_mask"].to(self.device),
                    batch_data["target_ids"].to(self.device),
                    batch_data["target_mask"].to(self.device),
                )

                self.optimizer.zero_grad()
                loss, _, _ = self.model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,
                                   target_mask=target_mask)

                # loss = loss.mean()
                # loss.backward(retain_graph=True)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                # print(loss.item())
                train_loss += loss.item() / len(self.train_loader)

                print(
                    f"{step}/{len(self.train_loader) // self.train_loader.batch_size}, train_loss: {loss.item():.4f}")

            print(f"epoch {epoch + 1} average loss: {train_loss:.4f}")
            self.epoch_log["train_loss"] = train_loss

            # validation

            self.model.eval()
            with torch.no_grad():

                val_loss = 0
                val_step = 0

                for val_data in self.val_loader:
                    val_step += 1

                    source_ids_val, source_mask_val, target_ids_val, target_mask_val = (
                        val_data["source_ids"].to(self.device),
                        val_data["source_mask"].to(self.device),
                        val_data["target_ids"].to(self.device),
                        val_data["target_mask"].to(self.device),
                    )
                    val_loss_step, _, _ = self.model(source_ids=source_ids_val, source_mask=source_mask_val, target_ids=target_ids_val,
                                            target_mask=target_mask_val)

                    val_loss_step =  val_loss_step.mean()
                    val_loss += val_loss_step.item() / len(self.val_loader)


                print(f"epoch {epoch + 1} average val loss: {val_loss:.4f}")

                self.epoch_log["val_loss"] = val_loss

            # update Logger File
            self.writeCSVLoggerFile()

            # save model checkpoint
            self.save_model_checkpoint()
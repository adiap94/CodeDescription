import os
import torch
import pandas as pd
import numpy as np
import logging
from data_loader import read_examples
import bleu
from bleu import *

logger = logging.getLogger(__name__)


class train_class():

    def __init__(self, model, optimizer, train_loader, val_loader,epoch_num, out_dir, device,scheduler,tokenizer,path_to_dev):
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
        self.csvLoggerFile_belu_path = os.path.join(out_dir, "belu_history.csv")
        self.best_belu_metric = np.inf
        self.dev_blue = 0
        self.step = 0
        self.tokenizer = tokenizer
        self.path_to_dev = path_to_dev
        self.belu_log = {}
        self.eval_examples = read_examples(self.path_to_dev)
        self.index = 0

    def writeCSVLoggerFile(self,belu=False):
        if not belu:
            df = pd.DataFrame([self.epoch_log])
            with open(self.csvLoggerFile_path, 'a') as f:
                df.to_csv(f, mode='a', header=f.tell() == 0, index=False)
        else:
            df2 = pd.DataFrame([self.belu_log])
            with open(self.csvLoggerFile_belu_path, 'a') as f:
                df2.to_csv(f, mode='a', header=f.tell() == 0, index=False)



    def save_model_checkpoint(self,belu = False):


        if not belu:
            PATH = os.path.join(self.out_dir, "Models", "last_model.pt")
            torch.save(self.model, PATH)
            print("saved last model")
            if self.epoch_log["val_loss"] < self.best_metric:
                self.best_metric = self.epoch_log["val_loss"]
                PATH = os.path.join(self.out_dir, "Models", "best_metric_model.pt")
                torch.save(self.model, PATH)
                print("saved new best metric model")
        else:
            PATH = os.path.join(self.out_dir, "Models", "Best_Bule_model.pt")
            torch.save(self.model, PATH)
            print("saved new best BLUE score model")
            # if self.epoch_log["belu_loss"] < self.best_belu_metric:
            #     self.best_metric = self.epoch_log["belu_loss"]
            #     PATH = os.path.join(self.out_dir, "Models", "best_belu_metric_model.pt")
            #     torch.save(self.model, PATH)
            #     print("saved new best BELU metric model")


    def calc_Belu_score(self):
        print("Starting BLEU evluation")
        self.model.eval()
        p = []


        max_example = min(500, len(self.eval_examples))
        tot_val = 0
        for val_data in self.val_loader:

            source_ids_val, source_mask_val = (
                val_data["source_ids"].to(self.device),
                val_data["source_mask"].to(self.device),
            )
            tot_val += source_ids_val.shape[0]
            if tot_val > max_example:
                break
            with torch.no_grad():
                preds = self.model(source_ids=source_ids_val, source_mask=source_mask_val)
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = self.tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    p.append(text)


        predictions = []
        print("Calculting BLEU score")
        with open(os.path.join(self.out_dir, "dev.output"), 'w') as f, open(
                os.path.join(self.out_dir, "dev.gold"), 'w') as f1:
            for ref, gold in zip(p[:max_example], self.eval_examples[:max_example]):
                predictions.append(str(gold.idx) + '\t' + ref)
                f.write(str(gold.idx) + '\t' + ref + '\n')
                f1.write(str(gold.idx) + '\t' + gold.target + '\n')
        (goldMap, predictionMap) = computeMaps(predictions,
                                                    os.path.join(self.out_dir, "dev.gold"))
        dev_bleu = round(bleuFromMaps(goldMap, predictionMap)[0], 2)
        logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
        logger.info("  " + "*" * 20)
        if dev_bleu > self.best_belu_metric:
            logger.info("  Best bleu:%s", dev_bleu)
            logger.info("  " + "*" * 20)
            self.best_belu_metric = dev_bleu
        self.belu_log["index"] = self.index
        self.belu_log["step"] = self.step
        self.belu_log["dev_bleu"] = dev_bleu
        self.index += 1
        self.save_model_checkpoint(True)
        self.writeCSVLoggerFile(True)
        self.model.train()



    def calc_val_loss(self,epoch):
        self.model.eval()
        with torch.no_grad():

            val_loss = 0
            tokens_num = 0
            val_step = 0
            eval_loss = 0

            for val_data in self.val_loader:
                val_step += 1

                source_ids_val, source_mask_val, target_ids_val, target_mask_val = (
                    val_data["source_ids"].to(self.device),
                    val_data["source_mask"].to(self.device),
                    val_data["target_ids"].to(self.device),
                    val_data["target_mask"].to(self.device),
                )
                _, val_loss_step, num = self.model(source_ids=source_ids_val, source_mask=source_mask_val,
                                                   target_ids=target_ids_val,
                                                   target_mask=target_mask_val)

                # val_loss_step =  val_loss_step.mean()
                val_loss += val_loss_step.sum().item()
                tokens_num += num.sum().item()
            eval_loss = val_loss / tokens_num
            result = {'eval_ppl': round(np.exp(eval_loss), 5),
                      'global_step': self.step + 1,}
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
            logger.info("  " + "*" * 20)
            print(f"step  {self.step + 1} , average val loss: {eval_loss:.4f}")

            self.epoch_log["epoch"] = epoch
            self.epoch_log["step"] = self.step
            self.epoch_log["val_loss"] = eval_loss
            self.epoch_log["eval_ppl"] = round(np.exp(eval_loss), 5)
            # update Logger File
            self.writeCSVLoggerFile()

            # save model checkpoint
            self.save_model_checkpoint()
            self.model.train()

    def train(self):

        for epoch in range(self.epoch_num):
            #self.epoch_log = {}
            self.epoch_log["epoch"] = epoch
            print(f"epoch {epoch}/{self.epoch_num}")

            self.model.train()

            train_loss = 0
            step = 0
            for batch_data in self.train_loader:
                step += 1
                self.step +=1
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

                if self.step%500 ==0:
                    self.calc_val_loss(epoch)

                if self.step%5000 ==0:
                    self.calc_Belu_score()

            print(f"epoch {epoch + 1} average loss: {train_loss:.4f}")
            self.epoch_log["train_loss"] = train_loss

            # validation

            self.model.eval()
            # with torch.no_grad():
            #
            #     val_loss = 0
            #     tokens_num = 0
            #     val_step = 0
            #     eval_loss = 0
            #
            #     for val_data in self.val_loader:
            #         val_step += 1
            #
            #         source_ids_val, source_mask_val, target_ids_val, target_mask_val = (
            #             val_data["source_ids"].to(self.device),
            #             val_data["source_mask"].to(self.device),
            #             val_data["target_ids"].to(self.device),
            #             val_data["target_mask"].to(self.device),
            #         )
            #         _, val_loss_step, num = self.model(source_ids=source_ids_val, source_mask=source_mask_val, target_ids=target_ids_val,
            #                                 target_mask=target_mask_val)
            #
            #         # val_loss_step =  val_loss_step.mean()
            #         val_loss += val_loss_step.sum().item()
            #         tokens_num += num.sum().item()
            #     eval_loss = val_loss / tokens_num
            #     result = {'eval_ppl': round(np.exp(eval_loss),5),
            #               'global_step': self.step+1,
            #               'train_loss': round(train_loss,5)}
            #     for key in sorted(result.keys()):
            #         logger.info("  %s = %s", key, str(result[key]))
            #     logger.info("  "+"*"*20)
            #     print(f"epoch {epoch+1} , average val loss: {eval_loss:.4f}")
            #
            #     self.epoch_log["epoch"] = epoch
            #     self.epoch_log["step"] = self.step
            #     self.epoch_log["val_loss"] = eval_loss
            #     self.epoch_log["eval_ppl"] = round(np.exp(eval_loss),5)
            #     self.calc_Belu_score()
            # update Logger File
            self.calc_val_loss(epoch)
            self.writeCSVLoggerFile()

            # save model checkpoint
            self.save_model_checkpoint()
            self.calc_Belu_score()




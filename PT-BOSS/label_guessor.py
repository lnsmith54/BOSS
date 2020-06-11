import torch
import math
import numpy as np

class LabelGuessor(object):

    def __init__(self, thresh):
        self.thresh = thresh

    def __call__(self, model, ims, balance, delT):
        org_state = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
        }
        is_train = model.training
        with torch.no_grad():
            model.train()
            all_probs = []
            logits = model(ims)
            probs = torch.softmax(logits, dim=1)
            scores, lbs = torch.max(probs, dim=1)
#            print("lbs ", lbs)
#            print("scores ", scores)
            mask = torch.ones_like(lbs,dtype=torch.float)
            labels, counts = np.unique(lbs.cpu(),return_counts=True)
#            print("labels", labels)
#            print("counts ", counts)
#            count_unlabels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#            for i in range(len(labels)):
#                count_unlabels[labels[i]] = counts[i]
            mxCount = max(counts)
            stdClass = np.std(counts)
#            print("stdClass ", stdClass)
            if balance > 0:
                if balance == 1 or balance == 4:
                    idx = (mask == 0.0)
                else:
                    idx = (scores > self.thresh)
                    delT = 0
                if mxCount > 0:
                    ratios =  [x/mxCount for x in counts]
                    for i in range(len(labels)):
                        tmp = (scores*(lbs==labels[i]).float()).ge(self.thresh - delT*(1-ratios[i])) # Which elements
                        idx = idx | tmp
                    if balance > 2:
                        labels, counts = np.unique(lbs[idx].cpu(),return_counts=True)
                    ratio = torch.zeros_like(mask,dtype=torch.float)
                    for i in range(len(labels)):
                        ratio += ((1/counts[i])*(lbs==labels[i]).float())  # Magnitude of mask elements
                    Z = torch.sum(mask[idx])
#                    print("ratio ",ratio)
                    mask = ratio[idx]
                    if Z > 0:
                        mask = Z * mask / torch.sum(mask)
                else:
                    idx = (scores > self.thresh)
                    mask[idx] = 1.0
            else:
                idx = scores > self.thresh
                mask = mask[idx]
            lbs = lbs[idx]
#            print("1. lbs ", lbs)
#            print("2. mask ", mask)

        model.load_state_dict(org_state)
        if is_train:
            model.train()
        else:
            model.eval()
        return lbs.detach(), idx, mask, stdClass


class EarlyStopping:
    def __init__(self, tolerance = 10, min_delta = 0.01):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.min_loss  = float('inf')
        self.counter   = 0
        self.early_stop = False
        self.model_dict = None
        self.optim_dict = None
        self.epoch = -1
        
    def __call__(self, val_loss, model_dict, optim_dict, epoch):
        if (self.min_loss - val_loss) > self.min_delta:
            self.counter = 0
            self.min_loss = val_loss
            self.model_dict = model_dict
            self.optim_dict = optim_dict
            self.epoch = epoch
        else:
            self.counter += 1
            if self.counter > self.tolerance:
                self.early_stop = True

def MLMLoss(encoder_output, target_ids, mask, criterion):
    logits = encoder_output[mask.bool()]
    masked_targets = target_ids[mask.bool()]
    loss = criterion(logits.view(-1, logits.size(-1)), 
                        masked_targets.view(-1))
    return loss
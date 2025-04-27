import torch
import gc
from skorch.callbacks import Callback 

# cleanup function for resource optimization
def cleanup(model, is_final=False):
    print("\nPerforming Cleanup...\n")

    # Clear CUDA cache 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if is_final:
            print("\u2713 -- FINAL: CUDA cache cleared.")
        else: 
            print("\u2713 -- CUDA cache cleared.")

    # run garbage collection
    gc.collect()
    if is_final:
        print("\u2713 -- FINAL: Garbage Collected.")
    else: 
        print("\u2713 -- Garbage Collected.")

    # delete the model when it is no longer needed
    if is_final and model:
        del model
        print("\u2713 -- FINAL: Model deleted.")

# callback to run with model.fit, will cleanup after each epoch
class EpochCleanupCallback(Callback):
    def on_epoch_end(self, net):
        cleanup(model=net, is_final=False) # passs current model for cleanup
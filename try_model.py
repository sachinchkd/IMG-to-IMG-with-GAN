from torch import nn, optim 
from torch.utils.data import DataLoader
from models.discriminator import Discriminator
from models.generator import Generator
import configs.config as config
from configs.utils import *
from configs.dataset import TestDataset 
def main():
    disc =Discriminator(in_channels=3).to(config.DEVICE) 
    gen =Generator(in_channels=3).to(config.DEVICE) 
    opt_disc = optim.Adam(disc.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.99))
    opt_gen = optim.Adam(gen.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.99))
    load_checkpoint("./data/gen.path.tar",gen,opt_gen,config.LEARNING_RATE)
    test_dataset = TestDataset(root_dir=config.TEST_DIR)

    test_loader = DataLoader(test_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS)

    save_some_examples(gen,test_loader,5,"results") 

if __name__ == "__main__":
    main()

from imports import *
from Preprocessing_PyTorch import *
from model import *

class Executing:
    """
    The Execution Class
    """
    def __init__(self):
        """
        Initializes the executing class.
        """
        self.batch_size = Config.BATCH_SIZE
        self.epochs = Config.NUM_EPOCHS
        self.lr = Config.LR
        self.metric = Config.METRIC

    def on_epoch_start(self, epoch):
        print(f'Epoch {epoch+1}/{self.epochs}')

    def prepare_batches(self):
        """
        Prepares the batches.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = self.df.get_data()
        self.X_train = torch.tensor(self.X_train, dtype=torch.long)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test, dtype=torch.long)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32)
        self.train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        self.test_dataset = torch.utils.data.TensorDataset(self.X_test, self.y_test)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, shuffle=True)

    def prepare_data(self):
        """
        Prepares the data.
        """
        start = time.time()
        self.df = Preprocessing()
        self.df.text2seq()
        tokenizer = self.df.get_tokenizer()
        self.input_size = len(tokenizer.word_index) + 1
        self.model = Classifier(self.input_size)
        self.prepare_batches()
        print("Done preparing data, done in {:.2f} seconds".format(time.time() - start))

    def fit(self):
        """
        Trains the model.
        """
        self.prepare_data()
        self.model.to(Config.DEVICE)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.history = []
        for epoch in range(0, self.epochs):
            start = time.time()    
            self.model.train()
            self.on_epoch_start(epoch)
            train_loss, train_acc, = 0, 0
            val_loss, val_acc = 0, 0
            for x, y in self.train_loader:
                trainSteps = len(self.train_loader.dataset) // self.batch_size
                x = x.to(Config.DEVICE)
                y = y.to(Config.DEVICE)
                y_pred = self.model(x)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss
                train_acc += self.metric(y_pred, y)
                with torch.no_grad():
                    self.model.eval()
                    for x_test, y_test in self.test_loader:
                        testSteps = len(self.test_loader.dataset) // self.batch_size
                        x_test = x_test.to(Config.DEVICE)
                        y_test = y_test.to(Config.DEVICE)
                        pred_test = self.model(x_test)
                        val_acc += self.metric(pred_test, y_test)
                        val_loss += criterion(pred_test, y_test)
            valCorrect = val_acc / len(self.test_loader.dataset)
            avgValLoss = val_loss / testSteps
            avgTrainLoss = train_loss / trainSteps
            trainCorrect = train_acc / len(self.train_loader.dataset)
            his = {'train_loss': avgTrainLoss, 'train_accuracy': trainCorrect, 'val_loss': avgValLoss, 'val_accuracy': valCorrect}
            self.history.append(his)
            self.on_epoch_end(his)
            print(f'Epoch {epoch+1} done in {time.time() - start:.2f} seconds')
            print("-"*10)

    def on_epoch_end(self, logs):
        """
        Prints the logs.
        """
        print(f'train_loss: {logs["train_loss"]:.2f}, train_accuracy: {logs["train_accuracy"]:.2f}')
        print(f'val_loss: {logs["val_loss"]:.2f}, val_accuracy: {logs["val_accuracy"]:.2f}')
        
        
    def get_history(self):
        """
        Returns the history.
        """
        return self.history

    def get_model(self):
        """
        Returns the model.
        """
        return self.model

if __name__ == "__main__":
    print("Start training process.")
    start = time.time()
    execute = Executing()
    execute.fit()
    end = time.time()
    print(f'Time taken for training: {end-start}.')
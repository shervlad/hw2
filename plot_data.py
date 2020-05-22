import matplotlib.pyplot as plt
import pickle

f = open('./logs/train_forward_logs.pickle', 'rb')
b = pickle.load(f)
plt.plot(b['train_loss'])
plt.ylabel('MSE loss')
plt.xlabel('epoch')
plt.title("final train loss=%s    test loss=%s"%(round(b['train_loss'][-1],4),round(b['test_loss'],4)))
plt.suptitle('Forward Model Training Loss')
plt.savefig('./plots/train_forward_model.png')
plt.close()

f = open('./logs/train_inverse_logs.pickle', 'rb')
b = pickle.load(f)
plt.plot(b['train_loss'])
plt.ylabel('MSE loss')
plt.xlabel('epoch')
plt.title("final train loss=%s    test loss=%s"%(round(b['train_loss'][-1],4),round(b['test_loss'],4)))
plt.suptitle('Inverse Model Training Loss')
plt.savefig('./plots/train_inverse_model.png')
plt.close()
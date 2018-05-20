import Bp
import loader
print "train start"


net = Bp.Network([2,3,1])
training_data = loader.load_data(1).transpose()
print training_data[0]
net.SGD(training_data, 30, 4, 3.0, test_data=training_data)
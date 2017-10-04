
import tensorflow as tf

W = tf.Variable([9.0], dtype = tf.float32)
b = tf.Variable([1.0], dtype = tf.float32)

sess = tf.Session()
init = tf.global_variables_initializer()

x = tf.placeholder(tf.float32)
with open('x1',"r") as f:
	x2 = []
	for line in f:
		x2.append(line)

y = tf.placeholder(tf.float32)

model = W*x+b

delta =  tf.square(model - y)
loss = tf.reduce_sum(delta)

sess.run(init)
#print(sess.run(loss, {x:[3.0], y:[8]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(1000):
  sess.run(train, {x: x2, y: [3.0, 6.0, 9.0, 12.0]})

print(sess.run([W, b]))


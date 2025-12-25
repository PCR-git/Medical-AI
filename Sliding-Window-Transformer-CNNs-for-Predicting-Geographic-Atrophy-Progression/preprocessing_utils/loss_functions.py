def weighted_cross_entropy(beta):
  def loss(y_true, y_pred):
    weight_a = beta * tf.cast(y_true, tf.float32)
    weight_b = 1 - tf.cast(y_true, tf.float32)
    
    o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b 
    return tf.reduce_mean(o)
  return loss

def dsc(y_true, y_pred):
     smooth = 1.
     y_true_f = keras.flatten(y_true)
     y_pred_f = keras.flatten(y_pred)
     intersection = keras.sum(y_true_f * y_pred_f)
     score = (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)
     return score

def dice_loss(y_true, y_pred):
    return (1 - dsc(y_true, y_pred)) + binary_crossentropy(y_true, y_pred)

def tversky_loss(beta):
  def loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = y_true * y_pred
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

    return 1 - tf.reduce_sum(numerator) / tf.reduce_sum(denominator)
  return loss


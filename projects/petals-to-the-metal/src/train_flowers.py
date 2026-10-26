# Original flower classifier training pipeline. MIT license.
import os, glob, json, re, time, math, random
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score

SEED = 20261023
SIZE = 331
BATCH = 64
CLASSES = 104
FINE_EPOCHS = 24
OUT = "/kaggle/working"
tf.keras.utils.set_random_seed(SEED)
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)
assert tf.config.list_physical_devices("GPU"), "Select GPU T4 x2 before running training."
tf.keras.mixed_precision.set_global_policy("mixed_float16")
strategy = tf.distribute.MirroredStrategy()
print("RUNTIME", tf.__version__, strategy.num_replicas_in_sync, flush=True)
roots = glob.glob("/kaggle/input/**/tfrecords-jpeg-331x331", recursive=True)
assert roots, "Attach the tpu-getting-started competition input."
ROOT = roots[0]
files = {s: sorted(glob.glob(f"{ROOT}/{s}/*.tfrec")) for s in ("train", "val", "test")}
assert all(files.values()), files

def count_records(paths):
    return sum(int(re.search(r"-(\d+)\.tfrec$", p).group(1)) for p in paths)

counts = {s:count_records(p) for s,p in files.items()}
print("COUNTS", counts, flush=True)
example = tf.train.Example.FromString(next(iter(tf.data.TFRecordDataset(files["train"][:1]))).numpy())
keys = example.features.feature.keys()
IMAGE_KEY = "image" if "image" in keys else "img"
LABEL_KEY = "class" if "class" in keys else "label"
print("SCHEMA", IMAGE_KEY, LABEL_KEY, list(keys), flush=True)

def parse_labelled(record):
    f = tf.io.parse_single_example(record, {IMAGE_KEY:tf.io.FixedLenFeature([],tf.string),LABEL_KEY:tf.io.FixedLenFeature([],tf.int64)})
    image = tf.io.decode_jpeg(f[IMAGE_KEY], channels=3)
    image = tf.ensure_shape(image, [SIZE,SIZE,3])
    return image, tf.cast(f[LABEL_KEY], tf.int32)

def parse_test(record):
    f = tf.io.parse_single_example(record, {IMAGE_KEY:tf.io.FixedLenFeature([],tf.string),"id":tf.io.FixedLenFeature([],tf.string)})
    return tf.ensure_shape(tf.io.decode_jpeg(f[IMAGE_KEY],channels=3),[SIZE,SIZE,3]),f["id"]

opts = tf.data.Options()
opts.threading.private_threadpool_size = 4
opts.experimental_deterministic = True
train_raw = tf.data.TFRecordDataset(files["train"],num_parallel_reads=4).with_options(opts).map(parse_labelled,num_parallel_calls=4).cache()
val_raw = tf.data.TFRecordDataset(files["val"]).with_options(opts).map(parse_labelled,num_parallel_calls=4).cache()
test_raw = tf.data.TFRecordDataset(files["test"]).with_options(opts).map(parse_test,num_parallel_calls=4).cache()

def augment(image,label):
    x = tf.cast(image, tf.float32)
    scale = tf.random.uniform([],0.78,1.0)
    side = tf.cast(scale * SIZE,tf.int32)
    x = tf.image.random_crop(x,[side,side,3])
    x = tf.image.resize(x,[SIZE,SIZE])
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.rot90(x,tf.random.uniform([],0,4,dtype=tf.int32))
    x = tf.image.random_brightness(x,20.0)
    x = tf.image.random_contrast(x,0.85,1.15)
    x = tf.image.random_saturation(x,0.85,1.15)
    return tf.clip_by_value(x,0.0,255.0),tf.one_hot(label,CLASSES)

def labelled(image,label):
    return tf.cast(image,tf.float32),tf.one_hot(label,CLASSES)

train = train_raw.shuffle(1024,seed=SEED,reshuffle_each_iteration=True).map(augment,num_parallel_calls=4).batch(BATCH,drop_remainder=True).prefetch(2)
train = train.apply(tf.data.experimental.assert_cardinality(counts["train"] // BATCH))
valid = val_raw.map(labelled,num_parallel_calls=4).batch(BATCH).prefetch(2)
valid = valid.apply(tf.data.experimental.assert_cardinality(math.ceil(counts["val"] / BATCH)))
yval = np.concatenate([y.numpy() for _,y in val_raw.batch(256)])
assert len(yval)==counts["val"]

with strategy.scope():
    backbone = tf.keras.applications.EfficientNetV2S(include_top=False,weights="imagenet",input_shape=(SIZE,SIZE,3),pooling="avg",include_preprocessing=True)
    backbone.trainable = False
    inp = tf.keras.Input((SIZE,SIZE,3))
    x = backbone(inp,training=False)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(CLASSES,activation="softmax",dtype="float32")(x)
    model = tf.keras.Model(inp,out)
    def compile_model(lr):
        optimizer = tf.keras.optimizers.Adam(lr)
        optimizer.build(model.trainable_variables)
        model.compile(optimizer=optimizer,loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy"),tf.keras.metrics.F1Score(average="macro",name="macro_f1")],jit_compile=False,steps_per_execution=1)
    compile_model(0.001)

print("WARMUP_START",flush=True)
h0 = model.fit(train,validation_data=valid,epochs=2,verbose=2)
with strategy.scope():
    backbone.trainable = True
    for layer in backbone.layers:
        if isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=False
    compile_model(0.0001)

class Progress(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        payload={"epoch":int(epoch+1),"metrics":{k:float(v) for k,v in (logs or {}).items()}}
        with open(f"{OUT}/progress.json","w") as f:json.dump(payload,f,indent=2)
        print("EPOCH_RESULT",json.dumps(payload),flush=True)

def schedule(epoch,lr):
    return 1e-6 + 0.5*(1e-4-1e-6)*(1+math.cos(math.pi*epoch/max(1,FINE_EPOCHS-1)))

callbacks=[tf.keras.callbacks.LearningRateScheduler(schedule),tf.keras.callbacks.ModelCheckpoint(f"{OUT}/best.weights.h5",monitor="val_macro_f1",mode="max",save_best_only=True,save_weights_only=True),tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1",mode="max",patience=6,restore_best_weights=True),tf.keras.callbacks.CSVLogger(f"{OUT}/training.csv"),Progress()]
print("FINETUNE_START",flush=True)
h1 = model.fit(train,validation_data=valid,epochs=FINE_EPOCHS,callbacks=callbacks,verbose=2)
model.load_weights(f"{OUT}/best.weights.h5")

# Deterministic prediction order preserves the official IDs for every TTA view.
def predict_views(raw,views=4):
    probs=[]
    for view in range(views):
        def view_map(x,y):
            x=tf.cast(x,tf.float32)
            if view==1:x=tf.image.flip_left_right(x)
            elif view==2:x=tf.image.flip_up_down(x)
            elif view==3:x=tf.image.rot90(x,2)
            return x
        ds=raw.map(view_map,num_parallel_calls=4).batch(BATCH).prefetch(2)
        probs.append(model.predict(ds,verbose=0))
    return np.mean(probs,axis=0)

print("VALIDATION_TTA_START",flush=True)
pval=predict_views(val_raw)
f1=float(f1_score(yval,pval.argmax(1),average="macro",labels=np.arange(CLASSES),zero_division=0))
acc=float(accuracy_score(yval,pval.argmax(1)))
print("VALIDATION_RESULT",json.dumps({"macro_f1":f1,"accuracy":acc}),flush=True)
np.savez_compressed(f"{OUT}/validation_predictions.npz",labels=yval,probabilities=pval)
ptest=predict_views(test_raw)
ids=np.concatenate([y.numpy() for _,y in test_raw.batch(256)]).astype(str)
assert len(ids)==counts["test"]==len(ptest)
assert len(set(ids))==len(ids) and np.isfinite(ptest).all()
samples=glob.glob("/kaggle/input/**/sample_submission.csv",recursive=True)
sample=pd.read_csv(samples[0],dtype={"id":str})
assert set(ids)==set(sample.id.astype(str))
pred=pd.Series(ptest.argmax(1),index=ids)
submission=pd.DataFrame({"id":sample.id,"label":pred.loc[sample.id].to_numpy()})
assert submission.label.between(0,CLASSES-1).all() and not submission.isna().any().any()
submission.to_csv(f"{OUT}/submission.csv",index=False)
np.savez_compressed(f"{OUT}/test_probabilities.npz",ids=ids,probabilities=ptest)
report={"status":"trained","model":"EfficientNetV2S ImageNet","seed":SEED,"size":SIZE,"counts":counts,"validation_macro_f1":f1,"validation_accuracy":acc,"fine_tune_epochs":len(h1.history["loss"]),"tta_views":4,"training_data":"official train split only; official val split held out","test_labels_used":False}
with open(f"{OUT}/validation_report.json","w") as f:json.dump(report,f,indent=2)
print("SUBMISSION_READY",len(submission),json.dumps(report),flush=True)

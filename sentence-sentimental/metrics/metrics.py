from datasets import load_metric

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = load_metric('accuracy').compute(predictions=preds, references=labels)['accuracy']
    f1 = load_metric('f1').compute(predictions=preds, references=labels, average='micro')['f1']

    return {'accuracy':acc, 'f1':f1}
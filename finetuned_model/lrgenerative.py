import types
from finetuned_model.basemodel import BaseModel
from transformers import AutoConfig,AutoTokenizer,AutoModelForSeq2SeqLM,get_linear_schedule_with_warmup



class LRGenerative(BaseModel):
    def __init__(self, arch='t5_large', train_batch_size=16, eval_batch_size=16, accumulate_grad_batches=1, learning_rate=1e-5, max_epochs=5,\
                    optimizer='adamw', adam_epsilon=1e-8, weight_decay=0.0, lr_scheduler='linear_with_warmup', warmup_updates=0.0, freeze_epochs=-1, gpus=1,\
                    hf_name='t5-large',save_dir=None, random_init=False):
        super().__init__(train_batch_size=train_batch_size, max_epochs=max_epochs, gpus=gpus)

        self.save_hyperparameters()

        self.p                         = types.SimpleNamespace()
        self.p.arch                    = arch
        self.p.train_batch_size        = train_batch_size
        self.p.eval_batch_size         = eval_batch_size
        self.p.accumulate_grad_batches = accumulate_grad_batches
        self.p.learning_rate           = learning_rate
        self.p.max_epochs              = max_epochs
        self.p.optimizer               = optimizer
        self.p.adam_epsilon            = adam_epsilon
        self.p.weight_decay            = weight_decay
        self.p.lr_scheduler            = lr_scheduler
        self.p.warmup_updates          = warmup_updates
        self.p.freeze_epochs           = freeze_epochs
        self.p.gpus                    = gpus
        self.p.save_dir                = save_dir
        self.p.hf_name    			   = hf_name

        self.reasoner  = AutoModelForSeq2SeqLM.from_pretrained(hf_name)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
        if random_init:
            model_config  = AutoConfig.from_pretrained(hf_name)
            self.reasoner = AutoModelForSeq2SeqLM(model_config)

        self.generator_options = {'min_length': 1, 'max_length': 128, 'num_beams': 1, 'num_return_sequences': 1, 'do_sample': False, 'top_k': 50, 'top_p': 1.0,\
                                    'temperature': 1.0, 'length_penalty': 1.0, 'repetition_penalty': 1.0}



    def forward(self, batch):
        outputs = self.reasoner(input_ids=batch['input_ids'], attention_mask=batch['input_attn'], labels=batch['output_ids'])
        return outputs


    def predict(self, batch):
        output_ids = self.reasoner.generate(input_ids=batch['input_ids'], attention_mask=batch['input_attn'], **self.generator_options)
        output_str = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return output_str


    def run_step(self, batch, split):
        if split == 'train' or split == 'dev':
            out  = self(batch)
            loss = out.loss
            self.log(f'{split}_loss_step', loss.item(), prog_bar=True)
            return {'loss': loss}

        else:
            preds     = self.predict(batch)
            return {'preds': preds}




    def aggregate_epoch(self, outputs, split):
        loss = torch.hstack([x['loss'] for x in outputs]).mean()
        if split == 'train' or split == 'dev':
            self.log(f'{split}_loss_epoch', loss.item())

        else:
            preds   = torch.cat([x['preds'] for x in outputs])
            with open(os.path.join(self.p.save_dir,'output.csv'),'w') as f:
                writer = csv.writer(f)
                writer.writerows(preds)






    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params'      : [p for n, p in self.reasoner.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.p.weight_decay,
            },
            {
                'params'      : [p for n, p in self.reasoner.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]

        if self.p.optimizer == 'adamw':
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.p.learning_rate,eps=self.p.adam_epsilon)
        else:
            raise NotImplementedError

        if self.p.lr_scheduler == 'linear_with_warmup':
            if self.p.warmup_updates > 1.0:
                warmup_steps = int(self.p.warmup_updates)
            else:
                warmup_steps = int(self.total_steps * self.p.warmup_updates)
            print(f'\nTotal steps: {self.total_steps} with warmup steps: {warmup_steps}\n')

            # for order transformers
            scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=self.total_steps)
            scheduler = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        elif self.p.lr_scheduler == 'fixed':
            return [optimizer]
        else:
            raise NotImplementedError

        return [optimizer], [scheduler]

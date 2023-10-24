from transformers.trainer import *
from transformers import Seq2SeqTrainer
import torch
from sampler import BalanceClassSampler, DistributedSamplerWrapper

class Seq2SeqTrainerForTemporalPretraining(Seq2SeqTrainer):
    def init_temporal(
            self,
            losses_keys=None,
            downsample_non_temporal_data=False,
            temporal_labels=None,
    ):
        self.losses_keys = losses_keys

        if downsample_non_temporal_data and temporal_labels is None:
            raise ValueError("``downsample_non_temporal_data'' requires ``temporal_labels'' to be given. ")
        
        self.downsample_non_temporal_data = downsample_non_temporal_data
        self.temporal_labels = temporal_labels
        self.log_global_step = 0
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            if self.log_global_step != self.state.global_step and \
                (self.state.global_step % self.args.logging_steps == 0 or self.state.global_step == 1):
                self.log_global_step = self.state.global_step
                # loss_dict = {
                #     "global_step": self.state.global_step,
                # }
                loss_dict = dict()
                for key in self.losses_keys:
                    if key in outputs:
                        if outputs[key] is not None and (outputs[key] != 0).sum() > 0:
                            if outputs[key].dim() == 0:
                                loss_dict[key] = round(outputs[key].detach().mean().item(), 4)
                            else:
                                loss_dict[key] = round(outputs[key][outputs[key].nonzero()].view(-1).detach().mean().item(), 4)
                        else:
                            loss_dict[key] = round(0.0, 4)
                self.log(logs=loss_dict)

        return (loss, outputs) if return_outputs else loss
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_length:
            raise NotImplementedError("We do not use group_by_length for temporal-modeling. ")
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.world_size <= 1:
                if self.downsample_non_temporal_data:
                    return BalanceClassSampler(self.temporal_labels, mode="downsampling")
                else:
                    return RandomSampler(self.train_dataset, generator=generator)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                raise NotImplementedError("This set of code (i.e., BalanceClassSampler) is not tested on TPU. ")
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            else:
                if self.downsample_non_temporal_data:
                    return DistributedSamplerWrapper(BalanceClassSampler(self.temporal_labels, mode="downsampling"))
                else:
                    return DistributedSampler(
                        self.train_dataset,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,
                        seed=seed,
                        shuffle=True, # add shuffle
                    )
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = self.new_get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def new_get_optimizer_cls_and_kwargs(self, args: TrainingArguments) -> Tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.
        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.
        """

        # parse args.optim_args
        optim_args = {}
        if hasattr(args, "optim_args") and args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        if args.optim == OptimizerNames.ADAFACTOR:
            optimizer_cls = Adafactor
            # optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
            logger.info("Using adafactor with scale_parameter=True ! ")
            print("Using adafactor with scale_parameter=True ! ")
            optimizer_kwargs.update({"scale_parameter": True, "relative_step": False})
        elif args.optim == OptimizerNames.ADAMW_HF:
            from transformers.optimization import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim in [OptimizerNames.ADAMW_TORCH, OptimizerNames.ADAMW_TORCH_FUSED]:
            from torch.optim import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
            if args.optim == OptimizerNames.ADAMW_TORCH_FUSED:
                optimizer_kwargs.update({"fused": True})
        elif args.optim == OptimizerNames.ADAMW_TORCH_XLA:
            try:
                from torch_xla.amp.syncfree import AdamW

                optimizer_cls = AdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import syncfree AdamW from torch_xla.")
        elif args.optim == OptimizerNames.ADAMW_APEX_FUSED:
            try:
                from apex.optimizers import FusedAdam

                optimizer_cls = FusedAdam
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate apex FusedAdam but apex is not installed!")
        elif args.optim == OptimizerNames.ADAMW_BNB:
            try:
                from bitsandbytes.optim import Adam8bit

                optimizer_cls = Adam8bit
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate bnb Adam8bit but bnb is not installed!")
        elif args.optim == OptimizerNames.ADAMW_ANYPRECISION:
            try:
                from torchdistx.optimizers import AnyPrecisionAdamW

                optimizer_cls = AnyPrecisionAdamW
                optimizer_kwargs.update(adam_kwargs)

                # TODO Change dtypes back to M=FP32, Var = BF16, Kahan = False once they can be cast together in torchdistx.
                optimizer_kwargs.update(
                    {
                        "use_kahan_summation": strtobool(optim_args.get("use_kahan_summation", "False")),
                        "momentum_dtype": getattr(torch, optim_args.get("momentum_dtype", "float32")),
                        "variance_dtype": getattr(torch, optim_args.get("variance_dtype", "float32")),
                        "compensation_buffer_dtype": getattr(
                            torch, optim_args.get("compensation_buffer_dtype", "bfloat16")
                        ),
                    }
                )
            except ImportError:
                raise ValueError("Please install https://github.com/pytorch/torchdistx")
        elif args.optim == OptimizerNames.SGD:
            optimizer_cls = torch.optim.SGD
        elif args.optim == OptimizerNames.ADAGRAD:
            optimizer_cls = torch.optim.Adagrad
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs
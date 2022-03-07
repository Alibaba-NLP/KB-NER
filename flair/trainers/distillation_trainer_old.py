from .trainer import *
from flair.list_data import ListCorpus
import math
import random
import pdb
import copy
from flair.datasets import CoupleDataset
def get_corpus_lengths(train_data):
	return [len(corpus) for corpus in train_data]

def get_corpus_iterations(train_data, batch_size):
	corpus_lengths=get_corpus_lengths(train_data)
	return [math.ceil(corpus_length/float(batch_size)) for corpus_length in corpus_lengths]

def generate_training_order(train_data,batch_size,training_order=None):
	if training_order is None:
		corpus_iters=get_corpus_iterations(train_data,batch_size)
		training_order=[]
		for idx, iters in enumerate(corpus_iters):
			training_order=training_order+iters*[idx]
	random.shuffle(training_order)
	return training_order




class ModelDistiller(ModelTrainer):
	def __init__(
		self,
		student: flair.nn.Model,
		teachers: List[flair.nn.Model],
		corpus: ListCorpus,
		optimizer: torch.optim.Optimizer = SGD,
		epoch: int = 0,
		optimizer_state: dict = None,
		scheduler_state: dict = None,
		use_tensorboard: bool = False,
		config = None,
	):
		"""
		Initialize a model trainer
		:param model: The model that you want to train. The model should inherit from flair.nn.Model
		:param corpus: The dataset used to train the model, should be of type Corpus
		:param optimizer: The optimizer to use (typically SGD or Adam)
		:param epoch: The starting epoch (normally 0 but could be higher if you continue training model)
		:param optimizer_state: Optimizer state (necessary if continue training from checkpoint)
		:param scheduler_state: Scheduler state (necessary if continue training from checkpoint)
		:param use_tensorboard: If True, writes out tensorboard information
		"""
		assert len(teachers)==len(corpus.train_list), 'Training data and teachers should be the same length now!'
		self.model: flair.nn.Model = student
		self.corpus: ListCorpus = corpus
		self.corpus_teacher: ListCorpus = copy.deepcopy(corpus)
		# self.corpus_mixed_train: ListCorpus = [CoupleDataset(student_set,self.corpus_teacher.train_list[index]) for index,student_set in enumerate(self.corpus.train_list)]
		self.teachers: List[flair.nn.Model] = teachers
		for teacher in self.teachers: teacher.eval()
		self.optimizer: torch.optim.Optimizer = optimizer
		self.epoch: int = epoch
		self.scheduler_state: dict = scheduler_state
		self.optimizer_state: dict = optimizer_state
		self.use_tensorboard: bool = use_tensorboard
		self.config = config
	def train(
		self,
		base_path: Union[Path, str],
		learning_rate: float = 0.1,
		mini_batch_size: int = 32,
		eval_mini_batch_size: int = None,
		max_epochs: int = 100,
		anneal_factor: float = 0.5,
		patience: int = 3,
		min_learning_rate: float = 0.0001,
		train_with_dev: bool = False,
		monitor_train: bool = False,
		monitor_test: bool = False,
		embeddings_storage_mode: str = "cpu",
		checkpoint: bool = False,
		save_final_model: bool = True,
		anneal_with_restarts: bool = False,
		shuffle: bool = True,
		param_selection_mode: bool = False,
		num_workers: int = 4,
		sampler=None,
		use_amp: bool = False,
		amp_opt_level: str = "O1",
		**kwargs,
	) -> dict:
		"""
		Trains any class that implements the flair.nn.Model interface.
		:param base_path: Main path to which all output during training is logged and models are saved
		:param learning_rate: Initial learning rate
		:param mini_batch_size: Size of mini-batches during training
		:param eval_mini_batch_size: Size of mini-batches during evaluation
		:param max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.
		:param anneal_factor: The factor by which the learning rate is annealed
		:param patience: Patience is the number of epochs with no improvement the Trainer waits
		 until annealing the learning rate
		:param min_learning_rate: If the learning rate falls below this threshold, training terminates
		:param train_with_dev: If True, training is performed using both train+dev data
		:param monitor_train: If True, training data is evaluated at end of each epoch
		:param monitor_test: If True, test data is evaluated at end of each epoch
		:param embeddings_storage_mode: One of 'none' (all embeddings are deleted and freshly recomputed),
		'cpu' (embeddings are stored on CPU) or 'gpu' (embeddings are stored on GPU)
		:param checkpoint: If True, a full checkpoint is saved at end of each epoch
		:param save_final_model: If True, final model is saved
		:param anneal_with_restarts: If True, the last best model is restored when annealing the learning rate
		:param shuffle: If True, data is shuffled during training
		:param param_selection_mode: If True, testing is performed against dev data. Use this mode when doing
		parameter selection.
		:param num_workers: Number of workers in your data loader.
		:param sampler: You can pass a data sampler here for special sampling of data.
		:param kwargs: Other arguments for the Optimizer
		:return:
		"""

		if self.use_tensorboard:
			try:
				from torch.utils.tensorboard import SummaryWriter

				writer = SummaryWriter()
			except:
				log_line(log)
				log.warning(
					"ATTENTION! PyTorch >= 1.1.0 and pillow are required for TensorBoard support!"
				)
				log_line(log)
				self.use_tensorboard = False
				pass

		if use_amp:
			if sys.version_info < (3, 0):
				raise RuntimeError("Apex currently only supports Python 3. Aborting.")
			if amp is None:
				raise RuntimeError(
					"Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
					"to enable mixed-precision training."
				)

		if eval_mini_batch_size is None:
			eval_mini_batch_size = mini_batch_size

		# cast string to Path
		if type(base_path) is str:
			base_path = Path(base_path)

		log_handler = add_file_handler(log, base_path / "training.log")

		log_line(log)
		log.info(f'Model: "{self.model}"')
		log_line(log)
		log.info(f'Corpus: "{self.corpus}"')
		log_line(log)
		log.info("Parameters:")
		log.info(f' - learning_rate: "{learning_rate}"')
		log.info(f' - mini_batch_size: "{mini_batch_size}"')
		log.info(f' - patience: "{patience}"')
		log.info(f' - anneal_factor: "{anneal_factor}"')
		log.info(f' - max_epochs: "{max_epochs}"')
		log.info(f' - shuffle: "{shuffle}"')
		log.info(f' - train_with_dev: "{train_with_dev}"')
		log_line(log)
		log.info(f'Model training base path: "{base_path}"')
		log_line(log)
		log.info(f"Device: {flair.device}")
		log_line(log)
		log.info(f"Embeddings storage mode: {embeddings_storage_mode}")

		# determine what splits (train, dev, test) to evaluate and log
		if monitor_train:
			assert 0, 'monitor_train is not supported now!'            
		# if train_with_dev:
		# 	assert 0, 'train_with_dev is not supported now!'

		log_train = True if monitor_train else False
		log_test = (
			True
			if (not param_selection_mode and self.corpus.test and monitor_test)
			else False
		)
		log_dev = True if not train_with_dev else False

		# prepare loss logging file and set up header
		loss_txt = init_output_file(base_path, "loss.tsv")

		weight_extractor = WeightExtractor(base_path)

		optimizer: torch.optim.Optimizer = self.optimizer(
			self.model.parameters(), lr=learning_rate, **kwargs
		)
		if self.optimizer_state is not None:
			optimizer.load_state_dict(self.optimizer_state)

		if use_amp:
			self.model, optimizer = amp.initialize(
				self.model, optimizer, opt_level=amp_opt_level
			)

		# minimize training loss if training with dev data, else maximize dev score
		anneal_mode = "min" if train_with_dev else "max"

		scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
			optimizer,
			factor=anneal_factor,
			patience=patience,
			mode=anneal_mode,
			verbose=True,
		)

		if self.scheduler_state is not None:
			scheduler.load_state_dict(self.scheduler_state)

		# start from here, the train data is a list now
		train_data = self.corpus.train_list
		train_data_teacher = self.corpus_teacher.train_list
		# train_data = self.corpus_mixed
		# if training also uses dev data, include in training set
		if train_with_dev:
			train_data = [ConcatDataset([train, self.corpus.dev_list[index]]) for index, train in enumerate(self.corpus.train_list)]
			train_data_teacher = [ConcatDataset([train, self.corpus_teacher.dev_list[index]]) for index, train in enumerate(self.corpus_teacher.train_list)]
			# train_data = [ConcatDataset([train, self.corpus_mixed.dev_list[index]]) for index, train in self.corpus_mixed.train_list]
			# train_data_teacher = ConcatDataset([self.corpus_teacher.train, self.corpus_teacher.dev])
			# train_data = ConcatDataset([self.corpus_mixed.train, self.corpus_mixed.dev])
		coupled_train_data = [CoupleDataset(data,train_data_teacher[index]) for index, data in enumerate(train_data)]

		# if sampler is not None:
		# 	sampler = sampler(train_data)
		# 	shuffle = False

		dev_score_history = []
		dev_loss_history = []
		train_loss_history = []

		# At any point you can hit Ctrl + C to break out of training early.
		try:
			previous_learning_rate = learning_rate
			training_order = None
			for epoch in range(0 + self.epoch, max_epochs + self.epoch):
				log_line(log)

				# get new learning rate
				for group in optimizer.param_groups:
					learning_rate = group["lr"]

				# reload last best model if annealing with restarts is enabled
				if (
					learning_rate != previous_learning_rate
					and anneal_with_restarts
					and (base_path / "best-model.pt").exists()
				):
					log.info("resetting to best model")
					self.model.load(base_path / "best-model.pt")

				previous_learning_rate = learning_rate

				# stop training if learning rate becomes too small
				if learning_rate < min_learning_rate:
					log_line(log)
					log.info("learning rate too small - quitting training!")
					log_line(log)
					break
				training_order=generate_training_order(train_data, mini_batch_size,training_order)
				batch_loader = [DataLoader(
					train_corpus,
					batch_size=mini_batch_size,
					shuffle=shuffle,
					num_workers=num_workers,
					sampler=sampler,
				) for index, train_corpus in enumerate(coupled_train_data)]

				loader_iterators = [iter(dataloader) for dataloader in batch_loader]
				self.model.train()
				# TODO: check teacher parameters fixed and with eval() mode

				train_loss: float = 0

				seen_batches = 0
				total_number_of_batches = sum([len(loader) for loader in batch_loader])

				modulo = max(1, int(total_number_of_batches / 10))

				# process mini-batches
				batch_time = 0
				if self.teacher_annealing:
					interpolation=1-(epoch*self.anneal_factor)/100.0
					if interpolation<0:
						interpolation=0
				else:
					interpolation=self.interpolation
				log.info("Current loss interpolation: "+ str(interpolation))
				for batch_no, corpus_id in enumerate(training_order):
					batch = next(loader_iterators[corpus_id])
					student_input, teacher_input = zip(*batch)
					student_input=list(student_input)
					teacher_input=list(teacher_input)
					lengths1 = torch.Tensor([len(sentence.tokens) for sentence in teacher_input])
					lengths2 = torch.Tensor([len(sentence.tokens) for sentence in student_input])
					assert (lengths1==lengths2).all(), 'two batches are not equal!'
					teacher = self.teachers[corpus_id]
					start_time = time.time()
					# loss = self.model.forward_loss(batch, teacher)

					loss = self.model.simple_forward_distillation_loss(student_input, teacher_input, teacher, interpolation=interpolation)

					optimizer.zero_grad()
					# Backward
					if use_amp:
						with amp.scale_loss(loss, optimizer) as scaled_loss:
							scaled_loss.backward()
					else:
						loss.backward()
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
					optimizer.step()

					seen_batches += 1
					train_loss += loss.item()

					# depending on memory mode, embeddings are moved to CPU, GPU or deleted
					store_embeddings(student_input, embeddings_storage_mode)
					store_embeddings(teacher_input, embeddings_storage_mode)

					batch_time += time.time() - start_time
					if batch_no % modulo == 0:
						log.info(
							f"epoch {epoch + 1} - iter {batch_no}/{total_number_of_batches} - loss "
							f"{train_loss / seen_batches:.8f} - samples/sec: {mini_batch_size * modulo / batch_time:.2f}"
						)
						batch_time = 0
						iteration = epoch * total_number_of_batches + batch_no
						if not param_selection_mode:
							weight_extractor.extract_weights(
								self.model.state_dict(), iteration
							)

				train_loss /= seen_batches

				self.model.eval()

				log_line(log)
				log.info(
					f"EPOCH {epoch + 1} done: loss {train_loss:.4f} - lr {learning_rate:.4f}"
				)

				if self.use_tensorboard:
					writer.add_scalar("train_loss", train_loss, epoch + 1)

				# anneal against train loss if training with dev, otherwise anneal against dev score
				current_score = train_loss

				# evaluate on train / dev / test split depending on training settings
				result_line: str = ""

				if log_train:
					train_eval_result, train_loss = self.model.evaluate(
						DataLoader(
							self.corpus.train,
							batch_size=eval_mini_batch_size,
							num_workers=num_workers,
						),
						embeddings_storage_mode=embeddings_storage_mode,
					)
					result_line += f"\t{train_eval_result.log_line}"

					# depending on memory mode, embeddings are moved to CPU, GPU or deleted
					store_embeddings(self.corpus.train, embeddings_storage_mode)

				if log_dev:
					# pdb.set_trace()
					dev_eval_result, dev_loss = self.model.evaluate(
						DataLoader(
							self.corpus.dev,
							batch_size=eval_mini_batch_size,
							num_workers=num_workers,
						),
						embeddings_storage_mode=embeddings_storage_mode,
					)
					result_line += f"\t{dev_loss}\t{dev_eval_result.log_line}"
					log.info(
						f"DEV : loss {dev_loss} - score {dev_eval_result.main_score}"
					)
					# calculate scores using dev data if available
					# append dev score to score history
					dev_score_history.append(dev_eval_result.main_score)
					dev_loss_history.append(dev_loss)

					current_score = dev_eval_result.main_score

					# depending on memory mode, embeddings are moved to CPU, GPU or deleted
					store_embeddings(self.corpus.dev, embeddings_storage_mode)

					if self.use_tensorboard:
						writer.add_scalar("dev_loss", dev_loss, epoch + 1)
						writer.add_scalar(
							"dev_score", dev_eval_result.main_score, epoch + 1
						)

				if log_test:
					test_eval_result, test_loss = self.model.evaluate(
						DataLoader(
							self.corpus.test,
							batch_size=eval_mini_batch_size,
							num_workers=num_workers,
						),
						base_path / "test.tsv",
						embeddings_storage_mode=embeddings_storage_mode,
					)
					result_line += f"\t{test_loss}\t{test_eval_result.log_line}"
					log.info(
						f"TEST : loss {test_loss} - score {test_eval_result.main_score}"
					)

					# depending on memory mode, embeddings are moved to CPU, GPU or deleted
					store_embeddings(self.corpus.test, embeddings_storage_mode)

					if self.use_tensorboard:
						writer.add_scalar("test_loss", test_loss, epoch + 1)
						writer.add_scalar(
							"test_score", test_eval_result.main_score, epoch + 1
						)

				# determine learning rate annealing through scheduler
				scheduler.step(current_score)

				train_loss_history.append(train_loss)

				# determine bad epoch number
				try:
					bad_epochs = scheduler.num_bad_epochs
				except:
					bad_epochs = 0
				for group in optimizer.param_groups:
					new_learning_rate = group["lr"]
				if new_learning_rate != previous_learning_rate:
					bad_epochs = patience + 1

				# log bad epochs
				log.info(f"BAD EPOCHS (no improvement): {bad_epochs}")

				# output log file
				with open(loss_txt, "a") as f:

					# make headers on first epoch
					if epoch == 0:
						f.write(
							f"EPOCH\tTIMESTAMP\tBAD_EPOCHS\tLEARNING_RATE\tTRAIN_LOSS"
						)

						if log_train:
							f.write(
								"\tTRAIN_"
								+ "\tTRAIN_".join(
									train_eval_result.log_header.split("\t")
								)
							)
						if log_dev:
							f.write(
								"\tDEV_LOSS\tDEV_"
								+ "\tDEV_".join(dev_eval_result.log_header.split("\t"))
							)
						if log_test:
							f.write(
								"\tTEST_LOSS\tTEST_"
								+ "\tTEST_".join(
									test_eval_result.log_header.split("\t")
								)
							)

					f.write(
						f"\n{epoch}\t{datetime.datetime.now():%H:%M:%S}\t{bad_epochs}\t{learning_rate:.4f}\t{train_loss}"
					)
					f.write(result_line)

				# if checkpoint is enable, save model at each epoch
				if checkpoint and not param_selection_mode:
					self.model.save_checkpoint(
						base_path / "checkpoint.pt",
						optimizer.state_dict(),
						scheduler.state_dict(),
						epoch + 1,
						train_loss,
					)

				# if we use dev data, remember best model based on dev evaluation score
				# pdb.set_trace()
				if (
					not train_with_dev
					and not param_selection_mode
					and current_score == scheduler.best
				):
					self.model.save(base_path / "best-model.pt")

			# if we do not use dev data for model selection, save final model
			if save_final_model and not param_selection_mode:
				self.model.save(base_path / "final-model.pt")

		except KeyboardInterrupt:
			log_line(log)
			log.info("Exiting from training early.")

			if self.use_tensorboard:
				writer.close()

			if not param_selection_mode:
				log.info("Saving model ...")
				self.model.save(base_path / "final-model.pt")
				log.info("Done.")

		# test best model if test data is present
		if self.corpus.test:
			final_score = self.final_test(base_path, eval_mini_batch_size, num_workers)
		else:
			final_score = 0
			log.info("Test data not provided setting final score to 0")

		log.removeHandler(log_handler)

		if self.use_tensorboard:
			writer.close()

		return {
			"test_score": final_score,
			"dev_score_history": dev_score_history,
			"train_loss_history": train_loss_history,
			"dev_loss_history": dev_loss_history,
		}
	@property
	def interpolation(self):
		try:
			return self.config['interpolation']
		except:
			return 0.5
	@property
	def teacher_annealing(self):
		try:
			return self.config['teacher_annealing']
		except:
			return False
	@property
	def anneal_factor(self):
		try:
			return self.config['anneal_factor']
		except:
			return 2
	
	
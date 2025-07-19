import wandb


class Logger:
    def __init__(self, experiment_name, project_name, no_logging=False, last_steps=-1,):
        self.total_steps = 0 if last_steps < 0 else last_steps
        self.running_loss = {}
        self.init_wandb = False
        self.no_logging = no_logging
        self.name = experiment_name
        self.sum_freq = 10

        if not self.no_logging and not self.init_wandb:
            self.init_wandb = True
            self._init_wandb(project_name)

    def _init_wandb(self, project_name):
        wandb.init(project=project_name, name=self.name)

    def log(self, d, step):
        if not self.no_logging:
            wandb.log(d, step=step)
        else:
            print(f'step {step}: {list(d.keys())[0]}: {list(d.values())[0]}')


    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/self.sum_freq for k in sorted(self.running_loss.keys())]
        
        if not self.no_logging and not self.init_wandb:
            self.init_wandb = True
            self._init_wandb()

        for k in self.running_loss:
            self.log({'Train/'+k: self.running_loss[k]/self.sum_freq}, step=self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.sum_freq == self.sum_freq-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if not self.no_logging and not self.init_wandb:
            self.init_wandb = True
            self._init_wandb()

        for key in results:
            self.log({'Eval/'+key: results[key]}, step=self.total_steps)

    def close(self):
        pass



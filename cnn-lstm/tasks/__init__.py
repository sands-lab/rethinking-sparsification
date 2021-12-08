def build(task_name, seed, device, timer, **kwargs):
    if task_name == "Cifar":
        from .cifar import CifarTask

        return CifarTask(
            seed=seed,
            device=device,
            timer=timer,
            architecture=kwargs.get("task_architecture", "ResNet18"),
        )

    if task_name == "Cifar100":
        from .cifar100 import Cifar100Task

        return Cifar100Task(
            seed=seed,
            device=device,
            timer=timer,
            architecture=kwargs.get("task_architecture", "resnet18"),
        )

    elif task_name == "LanguageModeling":
        from .language_modeling import LanguageModelingTask

        return LanguageModelingTask(
            seed=seed, device=device, timer=timer, batch_size=kwargs.get("optimizer_batch_size")
        )

    else:
        raise ValueError("Unknown task name")

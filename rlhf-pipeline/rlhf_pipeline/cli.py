from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from rlhf_pipeline.grpo_trainer import GRPOTrainer, GRPOTrainingConfig

app = typer.Typer(help="RLHF & GRPO Fine-Tuning Pipeline CLI")
console = Console()


@app.command()
def info():
    """Displays GRPO architecture and training hyperparameters."""
    config = GRPOTrainingConfig()
    console.print(Panel(f"[bold cyan]Model:[/bold cyan] {config.model_name}\n"
                        f"[bold cyan]Group Size (G):[/bold cyan] {config.group_size}\n"
                        f"[bold cyan]KL Coeff (Beta):[/bold cyan] {config.kl_coeff}\n"
                        f"[bold cyan]Clip Epsilon:[/bold cyan] {config.clip_eps}\n"
                        f"[bold cyan]Learning Rate:[/bold cyan] {config.learning_rate}",
                        title="GRPO Pipeline Configuration"))


@app.command()
def simulate_step(
    prompt: str = typer.Option("Solve for x: 2x + 4 = 10", "--prompt", "-p"),
    ground_truth: str = typer.Option("3", "--answer", "-a"),
):
    """Simulates a single GRPO training step with group completions and relative advantage calculation."""
    console.print(f"\n[bold cyan]🚀 Simulating GRPO Group Sampling & Relative Advantage Step...[/bold cyan]")
    console.print(f"Prompt: [yellow]{prompt}[/yellow] | Answer: [green]{ground_truth}[/green]\n")

    # Sample completions simulating varying reasoning quality & formatting
    sample_completions = [
        f"<think>2x + 4 = 10 -> 2x = 6 -> x = 3</think> <answer>{ground_truth}</answer>",  # Perfect format & answer
        f"The answer is {ground_truth}.",                                                 # Correct answer, missing tags
        f"<think>2x + 4 = 10 -> 2x = 8 -> x = 4</think> <answer>4</answer>",               # Bad reasoning & wrong answer
        f"<think>2x = 6 -> x = 3</think> <answer>{ground_truth}</answer>",               # Perfect format & answer
    ]

    trainer = GRPOTrainer(GRPOTrainingConfig(group_size=4))
    result = trainer.process_prompt_group(prompt, ground_truth, sample_completions)

    table = Table(title=f"GRPO Group Relative Evaluation (Group Mean Reward = {result.group_mean_reward:.2f})")
    table.add_column("Completion #", style="cyan")
    table.add_column("Completion Text", style="white")
    table.add_column("Reward (r_i)", style="magenta")
    table.add_column("Advantage (A_i)", style="green")

    for idx, (comp, rew, adv) in enumerate(zip(result.completions, result.rewards, result.advantages), 1):
        table.add_row(f"Sample #{idx}", comp[:65] + "...", f"{rew:.2f}", f"{adv:+.2f}")

    console.print(table)


if __name__ == "__main__":
    app()

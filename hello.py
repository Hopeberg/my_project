import typer

app = typer.Typer()


@app.command()
def Hello(count: int = 1):
    for x in range(count):
        typer.echo("Hello World!")


if __name__ == "__main__":
    app()

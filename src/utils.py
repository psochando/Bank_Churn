"""
Utilidades generales
"""
from rich.console import Console
from rich.theme import Theme
from rich import print, box
from rich.panel import Panel

consola = Console(
    theme=Theme(
        {
            "info": "cyan",
            "warning": "yellow",
            "error": "red",
            "success": "green",
            "debug": "blue",
            "important": "magenta",
            "log.time": 'salmon1'
        }
    ),
    log_path=False,
    tab_size=4
)

consola._log_render.omit_repeated_times = False


def intro(titulo, version, autor='Pablo S.', ancho=None):
    """
    Imprimir una intro de cabecera

    :param titulo: Título de la aplicación
    :param version: Versión de la aplicación
    :param autor:  Autor de la aplicación
    :param ancho: Ancho del título
    """

    pnl = Panel(
        f"[bright_cyan]{titulo}[/] [green]{version}[/]\n[bright_green]{autor}[/]",
        box=box.DOUBLE,
        width=ancho,
        expand=False
    )

    print(pnl)
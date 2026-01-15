#!/usr/bin/env python3
"""Semi-automatic SKOS alignment: search LOD, user selects."""

import re
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

console = Console()


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text)

PROVIDERS = {
    "wikidata": {"name": "Wikidata", "prefix": "wd"},
    "dbpedia": {"name": "DBpedia", "prefix": "dbr"},
    "schema": {"name": "Schema.org", "prefix": "schema"},
}


def search_wikidata(query: str, limit: int = 10, offset: int = 0) -> list[dict[str, str]]:
    try:
        r = requests.get("https://www.wikidata.org/w/api.php", params={
            "action": "wbsearchentities", "search": query, "language": "en",
            "format": "json", "limit": limit, "continue": offset
        }, timeout=10, headers={"User-Agent": "SmartphoneKG/1.0"})
        return [{"id": item["id"], "label": item.get("label", ""), "desc": item.get("description", "")}
                for item in r.json().get("search", [])]
    except Exception:
        return []


def search_dbpedia(query: str, limit: int = 10, offset: int = 0) -> list[dict[str, str]]:
    try:
        r = requests.get("https://lookup.dbpedia.org/api/search", params={
            "query": query, "maxResults": limit + offset, "format": "json"
        }, timeout=10, headers={"Accept": "application/json"})
        docs = r.json().get("docs", [])[offset:offset + limit]
        return [{"id": d["resource"][0].split("/")[-1], "label": d.get("label", [""])[0],
                 "desc": strip_html(d.get("comment", [""])[0])} for d in docs]
    except Exception:
        return []

SEARCH_FNS = {"wikidata": search_wikidata, "dbpedia": search_dbpedia}


def main():
    provider = "wikidata"
    alignments: list[tuple[str, str, str, str, str]] = []  # (concept, prefix, id, desc, match_type)

    console.print(Panel("[bold]SKOS Alignment Tool[/]", border_style="green"))

    while True:
        console.print()
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column(style="dim")
        table.add_row("c", "Align a concept")
        table.add_row("p", f"Provider [{PROVIDERS[provider]['name']}]")
        table.add_row("l", f"List [{len(alignments)}]")
        table.add_row("d", "Done")
        table.add_row("q", "Quit")
        console.print(table)

        cmd = Prompt.ask(">", default="").strip().lower()

        if cmd == 'q':
            break

        elif cmd == 'p':
            for key, p in PROVIDERS.items():
                console.print(f"  [cyan]{key}[/]  {p['name']}")
            choice = Prompt.ask(">", default=provider).strip().lower()
            if choice in PROVIDERS:
                provider = choice

        elif cmd == 'c':
            concept = Prompt.ask("[bold]Concept[/]", default="").strip()
            if not concept:
                continue

            query = concept
            offset = 0
            search_fn = SEARCH_FNS[provider]

            while True:
                with console.status(f"[cyan]Searching...[/]"):
                    results = search_fn(query, limit=5, offset=offset)

                if not results:
                    console.print("[yellow]No results[/]")
                    break

                console.print()
                console.print(f"[bold]{concept}[/] [dim]({query})[/]\n")
                for i, r in enumerate(results):
                    console.print(f"  [cyan]{i}[/]  [yellow]{r['id']}[/]  {r['label']}")
                    console.print(f"      [dim]{r['desc']}[/]\n")
                console.print("  [cyan]n[/] next  [cyan]r[/] new query  [cyan]s[/] skip  [cyan]0-4[/] select")

                choice = Prompt.ask(">", default="").strip().lower()

                if choice == 's':
                    break
                elif choice == 'n':
                    offset += 5
                elif choice == 'r':
                    query = Prompt.ask("Query", default=query).strip()
                    offset = 0
                elif choice.isdigit() and int(choice) < len(results):
                    selected = results[int(choice)]
                    console.print("  [cyan]e[/] exactMatch  [cyan]c[/] closeMatch")
                    match_choice = Prompt.ask(">", default="e").strip().lower()
                    match_type = "closeMatch" if match_choice == 'c' else "exactMatch"
                    prefix = PROVIDERS[provider]['prefix']
                    alignments.append((concept, prefix, selected['id'], selected['desc'], match_type))
                    console.print(f"[green]Added:[/] spv:{concept} skos:{match_type} {prefix}:{selected['id']}")
                    break

        elif cmd == 'l':
            if not alignments:
                console.print("[dim]No alignments[/]")
            else:
                table = Table(title="Alignments", show_header=True, header_style="bold")
                table.add_column("Concept")
                table.add_column("Match", style="cyan")
                table.add_column("Target", style="yellow")
                for concept, prefix, rid, desc, match_type in alignments:
                    table.add_row(f"spv:{concept}", match_type, f"{prefix}:{rid}")
                console.print(table)

        elif cmd == 'd':
            if not alignments:
                console.print("[dim]No alignments[/]")
                continue
            console.print("\n[bold]Turtle[/]\n")
            console.print("# region LOD ALIGNMENTS (semi-automatic)")
            for concept, prefix, rid, desc, match_type in alignments:
                console.print(f"spv:{concept} skos:{match_type} {prefix}:{rid} .")
                console.print(f"    # {desc}")
            console.print("# endregion")


if __name__ == "__main__":
    main()

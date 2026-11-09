"""Notebook を先頭から実行し、各セルの出力を書き戻す。"""

import contextlib
import io
import sys
import traceback
from pathlib import Path

import nbformat
from nbformat.v4 import new_output

DEFAULT_NOTEBOOK = Path(__file__).with_name("01_single_layer_pilot.ipynb")


def notebook_path(argument: str | None) -> Path:
    """引数がなければ、今までどおり同じディレクトリの予備実験ノートを返す。"""
    if argument is None:
        return DEFAULT_NOTEBOOK
    given = Path(argument)
    if given.is_file():
        return given.resolve()
    beside_script = Path(__file__).with_name(given.name)
    if beside_script.is_file():
        return beside_script.resolve()
    raise FileNotFoundError(f"Notebook が見つかりません: {argument}")


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    path = notebook_path(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"NOTEBOOK {path}", flush=True)
    notebook = nbformat.read(path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell_{index}", "exec")
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        print(f"\n===== CELL {index} =====", flush=True)
        captured = io.StringIO()
        stdout = Tee(sys.stdout, captured)
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(cell.source, f"cell_{index}", "exec"), namespace)
        except Exception as exc:
            text = captured.getvalue()
            cell.outputs = []
            cell.execution_count = index
            if text:
                cell.outputs.append(new_output(output_type="stream", name="stdout", text=text))
            cell.outputs.append(
                new_output(
                    output_type="error",
                    ename=type(exc).__name__,
                    evalue=str(exc),
                    traceback=traceback.format_exc().splitlines(),
                )
            )
            nbformat.write(notebook, path)
            print(traceback.format_exc(), flush=True)
            raise SystemExit(1) from exc
        text = captured.getvalue()
        cell.outputs = []
        cell.execution_count = index
        if text:
            cell.outputs.append(new_output(output_type="stream", name="stdout", text=text))
        nbformat.write(notebook, path)
    print("\nNOTEBOOK_FINISHED", flush=True)


if __name__ == "__main__":
    main()

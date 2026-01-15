from pathlib import Path
import threading
import logging
import queue

import soundfile
import gradio as gr
from audio_separator.separator import Separator


WORKSPACE_DIR = Path(__file__).parent.absolute()

# Configuration
MODEL_FILENAME = "vocals_mel_band_roformer.ckpt"
MODEL_DIR = WORKSPACE_DIR / "models"
BASE_OUTPUT_DIR = WORKSPACE_DIR / "output"


def ensure_dirs():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_separator():
    """Create a Separator instance and load the model. This is done once at startup."""
    sep = Separator(
        model_file_dir=str(MODEL_DIR),
        output_dir=str(BASE_OUTPUT_DIR),
        output_format="wav",
        output_single_stem="Vocals",
    )
    # ensure logger prints info/debug
    sep.logger.setLevel(logging.INFO)
    sep.load_model(model_filename=MODEL_FILENAME)
    return sep


ensure_dirs()


def separate_with_progress(audio_filepath: str):
    """Run separation in a background thread and stream logs + final file list.

    Yields tuples: (logs_text, files_list_or_None)
    """
    audio_path = Path(audio_filepath)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Uploaded file not found: {audio_filepath}")

    # Use a queue for real-time log streaming
    log_q: "queue.Queue[str]" = queue.Queue()

    class QueueHandler(logging.Handler):
        def __init__(self, q: "queue.Queue[str]"):
            super().__init__()
            self.q = q
            self.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
                )
            )

        def emit(self, record):
            try:
                msg = self.format(record)
                self.q.put(msg)
            except Exception:
                pass

    handler = QueueHandler(log_q)
    handler.setLevel(logging.DEBUG)

    # Attach handler to separator logger to capture runtime logs
    separator.logger.addHandler(handler)

    result: dict[str, list[str] | str | None] = {"files": None, "error": None}

    def worker():
        try:
            files = separator.separate(str(audio_path))
            result["files"] = files
        except Exception as e:
            result["error"] = str(e)
        finally:
            # signal completion
            log_q.put("__SEPARATION_DONE__")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    logs = []

    # Stream logs as they arrive in the queue
    while True:
        try:
            line = log_q.get(timeout=0.2)
        except queue.Empty:
            # if thread finished and queue empty break
            if not thread.is_alive() and log_q.empty():
                break
            continue

        if line == "__SEPARATION_DONE__":
            break

        logs.append(line)
        yield ("\n".join(logs), None)

    # Drain remaining messages
    while not log_q.empty():
        line = log_q.get()
        if line == "__SEPARATION_DONE__":
            continue
        logs.append(line)

    separator.logger.removeHandler(handler)

    full_logs = "\n".join(logs)

    if result.get("error"):
        full_logs += f"\nERROR: {result['error']}"
        yield (full_logs, None)
        return

    files = result.get("files") or []
    file_paths = [str((BASE_OUTPUT_DIR / fname).absolute()) for fname in files]

    yield (full_logs, file_paths)


def ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Audio Separator")
        gr.Markdown("## Input")
        with gr.Row():
            inp = gr.Audio(
                sources=["upload"], type="filepath", label="Upload/record audio"
            )

        btn = gr.Button("Separate", variant="primary")

        gr.Markdown("## Logs & Results")
        logs = gr.Textbox(label="Logs", lines=12)

        # Pre-create all possible output components, initially hidden
        MAX_AUDIO_NUM = 2
        output_audios = []
        for i in range(MAX_AUDIO_NUM):
            audio_comp = gr.Audio(
                label=f"Audio Output {i + 1}", visible=False, key=f"audio_{i}"
            )  # Key helps maintain state
            output_audios.append(audio_comp)

        def render_audios_and_clean(audio_paths: list[str]):
            updates = []
            for i in range(MAX_AUDIO_NUM):
                if i >= len(audio_paths):
                    updates.append(
                        gr.Audio(
                            label=f"Audio Output {i + 1}",
                            visible=False,
                            key=f"audio_{i}",
                        )
                    )
                    continue
                audio_path = audio_paths[i]
                data, sr = soundfile.read(audio_path)
                updates.append(
                    gr.Audio(
                        label=str(audio_path),
                        value=(sr, data),
                        visible=True,
                        key=f"audio_{i}",
                    )
                )
                # NOTE: clean audio
                Path(audio_path).unlink(missing_ok=True)
            return updates

        def _run_and_update(audio_path: str):
            for logs, fpaths in separate_with_progress(audio_path):
                if fpaths is None:
                    yield (logs, *[gr.Audio(visible=False)] * len(output_audios))
                else:
                    updates = render_audios_and_clean(fpaths)
                    yield (logs, *updates)

        btn.click(_run_and_update, inputs=inp, outputs=[logs, *output_audios])

    return demo


# Global separator (loads model into memory)
separator = make_separator()


if __name__ == "__main__":
    app = ui()
    # Listen on 0.0.0.0 as requested
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)

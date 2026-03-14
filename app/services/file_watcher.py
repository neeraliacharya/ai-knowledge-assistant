import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os

import app.services.rag_pipeline as rag_pipeline


class DocumentHandler(FileSystemEventHandler):

    def on_created(self, event):

        if event.src_path.lower().endswith(".pdf"):
            print("New document detected:", event.src_path)
            rag_pipeline.initialize_rag()


def start_watcher(folder="documents"):

    folder_path = os.path.abspath(folder)

    observer = Observer()
    handler = DocumentHandler()

    observer.schedule(handler, folder_path, recursive=False)
    observer.start()

    print("Document watcher started")
    print("Watching folder:", folder_path)

    return observer
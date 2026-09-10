"""
Interactive terminal chat about cluster health, recent incidents, and log
patterns. Uses the same OLLAMA_API_KEY / local Ollama chain as the
incident summarizer.

Usage:
    python chatbot/chat_cli.py --namespace default --label-selector app=checkout-worker
"""

import argparse

from chat_engine import ChatEngine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--namespace", default="default")
    p.add_argument("--label-selector", default=None, help="e.g. app=checkout-worker; omit to show all pods in the namespace")
    args = p.parse_args()

    engine = ChatEngine(namespace=args.namespace, label_selector=args.label_selector)

    print("Cluster health chatbot. Ask about recent incidents, pod status, or log patterns.")
    print("Type 'exit' or Ctrl+C to quit.\n")

    while True:
        try:
            question = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        answer, backend = engine.ask(question)
        print(f"\nassistant [{backend}]> {answer}\n")


if __name__ == "__main__":
    main()

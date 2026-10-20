"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


class ChatSession:
    """会話履歴を保持してLLMに質問するセッション。"""

    def __init__(self, model: str = MODEL) -> None:
        self.model = model
        self.client = genai.Client(api_key=get_api_key())
        self.chat = self.client.chats.create(model=self.model)

    def ask(self, question: str) -> str:
        response = self.chat.send_message(question)

        print(f"入力トークン数: {response.usage_metadata.prompt_token_count}")
        print(f"出力トークン数: {response.usage_metadata.candidates_token_count}")

        return response.text or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    session = ChatSession()
    question = args.question

    while True:
        print(f"\n[質問] {question}\n")
        try:
            print(session.ask(question))
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            raise SystemExit(1)

        try:
            next_question = input("\n[続けて質問（空Enter / quit / exit で終了）]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not next_question or next_question.lower() in {"quit", "exit"}:
            break
        question = next_question


if __name__ == "__main__":
    main()

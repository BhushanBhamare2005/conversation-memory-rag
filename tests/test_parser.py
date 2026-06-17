from app.ingestion.conversation_parser import split_conversation_turns


def test_split_conversation_turns_preserves_speakers():
    conversation = "User 1: Hello\nUser 2: Hi there\nUser 1: How are you?"
    turns = split_conversation_turns(conversation)
    assert turns[0][0] == "User 1"
    assert turns[1][0] == "User 2"
    assert turns[2][1] == "How are you?"

import addtwo

def test_add():
    result = addtwo.add(2, 3)
    expected = 5
    assert result == expected

    result = addtwo.add(-1, 1)
    expected = 0
    assert result == expected

    result = addtwo.add(0, 0)
    expected = 0
    assert result == expected

    result = addtwo.add(2.5, 3.5)
    expected = 6.0
    assert result == expected
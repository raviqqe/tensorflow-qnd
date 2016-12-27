import sys


def main():
    with open(sys.argv[1]) as f1, open(sys.argv[2]) as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    assert len(lines1) == len(lines2)

    print(sum(int(line1 == line2) for line1, line2 in zip(lines1, lines2))
          / len(lines1))


if __name__ == "__main__":
    main()

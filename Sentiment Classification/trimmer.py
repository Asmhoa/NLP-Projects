import sys

readingFile = open("test.txt", "r")
lines = []
writingFile = open("test_trimmed.txt", "w")

for line in readingFile:
    if len(line) > 250:
        lines.append(line)

writingFile.writelines(lines)
readingFile.close()
writingFile.close()
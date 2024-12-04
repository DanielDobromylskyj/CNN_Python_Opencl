

class File:
    def __init__(self, path):
        self.path = path
        self.segments = {}

    @staticmethod
    def __read_next_int(fb):
        bin_data = b""
        while True:
            char = fb.read(1)

            if char == b"":
                return

            if char == b"|":
                return int(bin_data.decode())

            bin_data += char

    def load(self):
        with open(self.path, "rb") as fb:
            while True:
                segment_name_size = self.__read_next_int(fb)

                if segment_name_size is None:
                    break

                segment_name = fb.read(segment_name_size).decode()

                segment_data_size = self.__read_next_int(fb)
                segment_data = fb.read(segment_data_size)

                self.segments[segment_name] = segment_data

    @staticmethod
    def __write_size_of_data(data):
        return str(len(data)).encode() + b"|"

    def __write_segment(self, segment_key):
        segment_data = self.segments[segment_key]

        if type(segment_data) is not bytes:
            segment_data = segment_data.encode()

        return self.__write_size_of_data(segment_key) + segment_key.encode() + self.__write_size_of_data(segment_data) + segment_data

    def write(self):
        file_binary = b""
        for segment_key in self.segments.keys():
            file_binary += self.__write_segment(segment_key)

        with open(self.path, "wb") as fb:
            fb.write(file_binary)


if __name__ == "__main__":
    f = File("test_file.pyn")
    f.segments = {"seg1": b"Some Binary Data", "seg2": b"Some More Data"}
    f.write()

    f.segments = {}
    f.load()
    print(f.segments)

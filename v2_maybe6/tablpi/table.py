
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def item_to_string(item):
    if type(item) in (int, float):
        return f" {bcolors.OKBLUE}{item}{bcolors.ENDC} "

    if type(item) is str:
        return f" {item} "

    if type(item) is bool:
        return f" {bcolors.OKGREEN if item else bcolors.FAIL}{'True' if item else 'False'}{bcolors.ENDC} "

    raise ValueError("Invalid item type:", type(item))

def print_table(row_names: list[str], col_names: list[str], data: list[list[int | float | str | bool]]) -> None:
    """ Must be in row-col (index is y, x)"""
    col_widths = [len(item_to_string(name)) for name in col_names]
    max_row_name_width = len(max(row_names, key=lambda x: len(x)))

    for row in data:
        for col_index, col in enumerate(row):
            sample_string = item_to_string(col)

            if len(sample_string) > col_widths[col_index]:
                col_widths[col_index] = len(sample_string)

    table_width = sum(col_widths) + len(col_widths) + 1

    def print_line():
        print("-" * (max_row_name_width + 1), end="")
        print("+", end="")

        for width in col_widths:
            print("-" * width, end="")
            print("+", end="")

        print()




    print_line()
    print(f"Table{' ' * (max_row_name_width - 5)} |")

    for row_index, row in enumerate(data):
        print_line()
        print(f"{row_names[row_index]}{' ' * (max_row_name_width - len(row_names[row_index]))} |")

    print_line()



if __name__ == "__main__":
    row = ["Sample A2", "Sample B41", "Sample C"]
    col = ["Test A", "Test B", "C", "41.3"]

    data = [
        [True, True, False, True],
        [False, True, False, True],
        [True, True, True, False],
    ]

    print_table(row, col, data)

#!/usr/bin/env python3


from termcolor import colored, cprint


print_user_msg = lambda msg: cprint(msg, "blue")

print_assistant_msg = lambda msg: cprint(msg, "green")

print_error_msg = lambda msg: cprint(msg, "red")

print_warning_msg = lambda msg: cprint(msg, "yellow")

print_info_msg = lambda msg: cprint(msg, "cyan")

print_debug_msg = lambda msg: cprint(msg, "magenta")

import os
import re
import requests
import ipaddress
from urllib.parse import urlparse


def  check_dir(self):
    # Validate path
    if self.path and not (os.path.exists(self.path) and os.path.isfile(self.path)):
        raise ValueError("Provided path is not a valid file")


def check_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def check_ip_address(ip_address):
    try:
        ipaddress.ip_address(ip_address)
        return True
    except ValueError:
        return False
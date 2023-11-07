"""
Utilities used for logging
"""

from json.decoder import JSONDecodeError

import requests

INSTANCE_INFO_URL = "http://169.254.169.254/latest/dynamic/instance-identity/document"
INSTANCE_REQUEST_TIMEOUT = 0.1


def get_instance_info() -> dict:
    """Provides information about a compute instance on which this code is being executed

    For now this method is designed only to collect info about AWS instances. For more info check:
    https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-identity-documents.html

    :return: A dictionary with information
    """
    try:
        response = requests.get(INSTANCE_INFO_URL, timeout=INSTANCE_REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {"error": "Not an AWS instance or AWS IMDSv1 service not available"}
    except JSONDecodeError:
        return {"error": f"Failed to decode a response from {INSTANCE_INFO_URL}"}

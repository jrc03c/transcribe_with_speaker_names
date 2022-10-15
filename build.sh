python setup.py sdist

echo ""

YELLOW="\033[1;33m"
RED="\033[1;31m"
CLEAR="\033[0m"

echo -e "${YELLOW}NOTE: Don't forget to update the version number in ${RED}setup.py${YELLOW} if you haven't already done so!${CLEAR}"
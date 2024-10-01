#!/bin/bash

# Function to convert the changelog
convert_changelog() {
    local file_path="$1"
    local formatted_changelog="+++\ntitle = \"Changelog\"\ndescription = \"Changelog\"\ndraft = false\nweight = 100\nsort_by = \"weight\"\ntemplate = \"releases/page.html\"\n\n[extra]\ntoc = true\ntop = false\nicon = \"\"\norder = 1\n+++\n\n"

    # Flag to skip the changelog header
    local skip_header=true

    # Read each line of the changelog file
    while IFS= read -r line; do
        # Skip lines until the first occurrence of "## unreleased"
        if $skip_header && ! [[ $line =~ ^##\ unreleased ]]; then
            continue
        fi
        skip_header=false

        # Check if the line starts with "## unreleased"
        if ! [[ $line =~ ^##\ unreleased ]]; then
            formatted_changelog+="$line\n"
        fi
    done < "$file_path"

    echo -e "$formatted_changelog"
}

# Check if a file path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <changelog_file>"
    exit 1
fi

# Convert the changelog
converted_changelog=$(convert_changelog "$1")
echo "$converted_changelog"

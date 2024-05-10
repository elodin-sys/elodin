#!/bin/bash

# Function to convert the changelog
convert_changelog() {
    local file_path="$1"
    local formatted_changelog="---\ntitle: Changelog\nicon: bars-staggered\n---\n\n"

    # Flag to skip the changelog header
    local skip_header=true

    # Flag to skip the unreleased section
    local skip_unreleased=false

    # Read each line of the changelog file
    while IFS= read -r line; do
        # Skip lines until the first occurrence of "## [unreleased]"
        if $skip_header && ! [[ $line =~ ^##\ \[unreleased\] ]]; then
            continue
        else
            skip_header=false
        fi

        # Check if the line starts with "## [unreleased]"
        if [[ $line =~ ^##\ \[unreleased\] ]]; then
            skip_unreleased=true
        # Check if the line starts with "## [vX.X.X]"
        elif [[ $line =~ ^##\ \[v[0-9]+\.[0-9]+\.[0-9]+\] ]]; then
            skip_unreleased=false
            # Remove links from the line
            line=$(echo "$line" | sed -E 's/\[v([0-9]+\.[0-9]+\.[0-9]+)\]/v\1/g')
            formatted_changelog+="\n$line\n"
        elif ! $skip_unreleased && [[ ! $line =~ \[[v0-9]+\.[0-9]+\.[0-9]+\]: ]] && [[ ! $line =~ ^\[unreleased\]: ]]; then
            # Exclude lines with links to GitHub comparisons and unreleased section
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


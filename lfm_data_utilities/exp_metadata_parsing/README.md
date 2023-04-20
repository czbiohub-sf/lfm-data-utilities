### exp_metadata_parser.py
Compile all experiment metadata into a table. A truncated table is displayed in the terminal for readability, and an untruncated text file is saved as `metadata_compilation.txt` in the working directory.

Parameters
- -f / --file: Folder containing experiment runs
- -v / --verbose (optional): Display all experiment metadata columns
- -i / --include (optional): Add specific metadata columns to be displayed

Default columns displayed: 
- directory
- notes
- git_branch

Complete list of columns that can be displayed (display all using `-v` or select some using `'i'`)
- directory
- filename
- operator_id
- participant_id
- flowcell_id
- target_flowrate
- site
- notes
- scope
- camera
- exposure
- target_brightness
- git_branch
- git_commit


#### Example usage
To compile metadata with default columns displayed:
`python3 -f <per image metadata file>`

Output:
```
                                directory                          notes git_branch
0    2023-03-06-153848\2023-03-06-154334_                  gametocytes!!    develop
1    2023-03-06-153848\2023-03-06-154640_                    gametocytes    develop
2    2023-03-06-153848\2023-03-06-155058_                    gametocytes    develop
...
```

To compile metadata with additional column displayed:
`python3 -f <per image metadata file> -i scope` or `python3 -f <per image metadata file> --include scope`

Output:
```
                                directory                          notes git_branch     scope
0    2023-03-06-153848\2023-03-06-154334_                  gametocytes!!    develop  lfm-ohmu
1    2023-03-06-153848\2023-03-06-154640_                    gametocytes    develop  lfm-ohmu
2    2023-03-06-153848\2023-03-06-155058_                    gametocytes    develop  lfm-ohmu
```

To compile metadata with all columns displayed:
`python3 -f <per image metadata file> -v` or `python3 -f <per image metadata file> --verbose`
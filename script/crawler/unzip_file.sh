directory=data/ddg

find "$directory" -type f -name "*.pdb.gz" -exec sh -c '
  for file do
    gunzip "$file" && echo "unzip $file"
  done
' sh {} +
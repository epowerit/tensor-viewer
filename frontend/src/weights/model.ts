export function checkpointFileIssue(file: {
  name: string;
  size: number;
}): string {
  if (!/\.(pt|pth)$/i.test(file.name))
    return "Choose a PyTorch .pt or .pth state dictionary.";
  if (file.name.length > 200)
    return "Use a filename of at most 200 characters.";
  if (!file.size) return "This file is empty.";
  if (file.size > 64 * 1024 * 1024)
    return "Use a checkpoint file up to 64 MiB.";
  return "";
}

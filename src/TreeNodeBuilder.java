import java.util.LinkedList;
import java.util.Queue;

public class TreeNodeBuilder {

    TreeNode root;
    public TreeNodeBuilder(Object[] list) {
        Queue<TreeNode> queue = new LinkedList<>();
        root = new TreeNode(Integer.parseInt(list[0].toString()));
        queue.add(root);
        int index = 1;
        while (!queue.isEmpty() && index < list.length) {
            TreeNode node = queue.poll();
            if (node == null) {
                index++;
                break;
            } else {
                node.left = build(list[index]);
                index++;
                if (index < list.length)
                    node.right = build(list[index]);
                index++;
                queue.add(node.left);
                queue.add(node.right);
            }
        }
    }

    public TreeNode getRoot() {
        return root;
    }

    private TreeNode build(Object obj) {
        if (obj == null)
            return null;
        else
            return new TreeNode(Integer.parseInt(obj.toString()));
    }
}

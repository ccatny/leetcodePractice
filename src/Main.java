import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        JavaUtilClass javaUtilClass = new JavaUtilClass();
        int[] nums = {2,3,1,5,6,4};
        List<List<String>> list = new ArrayList<List<String>>();
        int[][] candidates = new int[3][3];
        ListNode node = new ListNode(1, new ListNode(1, new ListNode(2, new ListNode(1))));
        TreeNode root = new TreeNode(6, new TreeNode(2, new TreeNode(2, new TreeNode(0), new TreeNode(4)), new TreeNode(8)), new TreeNode(3, null, new TreeNode(4)));
        for (int i =0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                candidates[i][j] = i*3 + j + 1;
        }
        List<List<Integer>> intList = new ArrayList<List<Integer>>();
        intList.add(List.of(2));
        intList.add(List.of(3, 4));
        intList.add(List.of(6, 5, 7));
        intList.add(List.of(4,1,8,3));
        Object[] objects = {3,9,20,null,null,15,7};
        TreeNodeBuilder treeNodeBuilder = new TreeNodeBuilder(objects);
        Node rootNode = new Node(1, new Node(2, new Node(4), new Node(5), null), new Node(3, new Node(6), new Node(7), null), null);
        ArrayList<String> arrayList = new ArrayList<>(Arrays.asList("hot","dot","dog","lot","log","cog"));
        Collections.sort(arrayList);
        javaUtilClass.singleNumber2(new int[]{2,1,2,3,4,1});
    }
}
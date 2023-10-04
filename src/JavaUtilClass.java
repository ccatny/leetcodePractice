import kotlin.Pair;

import java.util.*;

public class JavaUtilClass {
    public int countRangeSum(int[] nums, int lower, int upper) {
        int[] sums = new int[nums.length];
        int sum = 0;
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            sum = sum + nums[i];
            sums[i] = sum;
            if (sum >= lower && sum <= upper)
                count++;
        }
        for (int i = 1; i < nums.length; i++) {
            for (int j = i; j < nums.length; j++) {
                int cur = sums[j] - sums[i-1];
                if (cur>= lower && cur <= upper)
                    count++;
            }
        }
        return count;
    }

    public int minPatches(int[] nums, int n) {
        boolean[] arrive = new boolean[n];
        int count = 0;
        ArrayList<Integer> list = new ArrayList<>();
        for (int i = 0; i < nums.length; i++)
            list.add(nums[i]);
        for (int i = 0; i < n; i++) {
            if(!reachable(list, arrive, i+1)) {
                add(list, i+1);
                count++;
            }
            arrive[i] = true;
        }
        return count;
    }

    private boolean reachable(ArrayList<Integer> list, boolean[] arrive, int num) {
        int i = 0;
        while (i < list.size() && list.get(i) < num / 2) {
            if (arrive[num - list.get(i) - 1])
                return true;
            i++;
        }
        while (i < list.size() && list.get(i) <= num) {
            if (list.get(i) == num)
                return true;
            i++;
        }
        return false;
    }

    private void add(ArrayList<Integer> list, int num) {
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) > num) {
                list.add(i, num);
                return;
            }
        }
        list.add(num);
    }

    public List<String> findItinerary(List<List<String>> tickets) {
        TreeMap<String, List<String>> map = new TreeMap<>();
        for (int i = 0; i < tickets.size(); i++) {
            List<String> ticket = tickets.get(i);
            if (!map.containsKey(ticket.get(0))) {
                List<String> list = new ArrayList<String>();
                list.add(ticket.get(1));
                map.put(ticket.get(0), list);
            } else {
                map.get(ticket.get(0)).add(ticket.get(1));
                Collections.sort(map.get(ticket.get(0)));
            }
        }
        List<String> answer = new LinkedList<>();
        search(map, "JFK", answer);
        Collections.reverse(answer);
        return answer;
    }

    private void search(TreeMap<String, List<String>> map, String cur, List<String> answer) {
        List<String> list = map.get(cur);
        while (list != null && list.size() > 0) {
            String s = list.get(0);
            list.remove(0);
            search(map, s, answer);
        }
        answer.add(cur);
    }

    public boolean increasingTriplet(int[] nums) {
        Deque<Integer> stack = new LinkedList<>();
        int pivot = nums[0];
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < pivot || i == 0) {
                pivot = nums[i];
                for (int j = i + 1; j < nums.length; j++) {
                    if (nums[j] > pivot) {
                        stack.push(nums[j]);
                        pivot = nums[j];
                        if (stack.size() == 2)
                            return true;
                    }
                }
                stack.clear();
            }
        }
        return false;
    }

    public List<List<Integer>> select(int k, int[][] candiadtes) {
        List<List<Integer>> answer = new ArrayList<List<Integer>>();
        add(answer, new ArrayList<>(), k, candiadtes, new int[k]);
        answer.size();
        return answer;
    }

    private void add(List<List<Integer>> answer, List<Integer> temp, int k, int[][] candidate, int[] index) {
        if (temp.size() == k) {
            answer.add(new ArrayList<>(temp));
        } else {
            for (int i = 0; i < k; i++) {
                int l = temp.size();
                temp.add(candidate[l][i]);
                add(answer, temp, k, candidate, index);
                temp.remove(temp.size() - 1);
            }
        }
    }

    public String caesarCipher(String s, int k) {
        // Write your code here
        StringBuilder builder = new StringBuilder();
      
        for (int i = 0; i < s.length(); i++) {
            char temp = s.charAt(i);
            int ascii = (int)temp;
            if (temp >= 'A' && temp <= 'Z') {
                ascii = (ascii + k) > (int)'Z' ? (ascii + k - (int)'Z') % 26 + (int)'A' - 1: ascii + k;
                builder.append((char) ascii);
                System.out.println("c : " + ascii + " A-Z ");
            }
            else if ((temp >= 'a' && temp <= 'z')) {
                ascii = (ascii + k) > (int)'z' ? (ascii + k - (int)'z') % 26 + (int)'a' - 1: ascii + k;
                builder.append((char) ascii);
                System.out.println("c : " + ascii + " a-z ");
            }
            else {
                System.out.println("c : " + temp);
                builder.append(temp);
            }
        }
        return builder.toString();
    }

    public void ptest(int n) {
        if (n <= 1) {
            System.out.println(n);
        }
        else {
            ptest(n / 3);
            System.out.println(n % 3);
        }
    }

    public void separateNumbers(String s) {
        // Write your code here
        int i = 0, j = 1;
        int maxStep = s.length() / 2;
        int r = 1;
        for (r = 1; r <= maxStep; r++) {
            boolean increase = false;
            j = i + r;
            while (j + r <= s.length() && s.charAt(i) != '0' && s.charAt(j) != 0) {
                long pre = Long.parseLong(s.substring(i, j));
                long next = Long.parseLong(s.substring(j, j+r ));
                if (next - pre == 1) {
                    i = i + r;
                    j = j + r;
                }
                else if (j + r + 1 <= s.length() && !increase) {
                    next = Long.parseLong(s.substring(j, j+r+1));
                    if (next - pre == 1) {
                        i = i + r;
                        j = j + r + 1;
                        r++;
                        increase = true;
                    }
                    else
                        break;
                }
                else {
                    break;
                }
            }
            if (i + r == s.length()) {
                if (increase)
                    r--;
                System.out.println("YES " + s.substring(0, r));
                return;
            }
            if (increase)
                r--;
        }
        System.out.println("NO");
    }

    public String minWindow(String s, String t) {
        HashMap<Character, Integer> map = new HashMap<>();
        int[] origin = new int[58];
        String answer = "";
        for (int i = 0; i < t.length(); i++) {
            map.put(t.charAt(i), map.getOrDefault(t.charAt(i), 0) + 1);
            origin[t.charAt(i)-'A']++;
        }
        int i = 0, j = 0;
        int[] count = new int[58];
        int countNum = 0;
        while (j < s.length()) {
            if (map.containsKey(s.charAt(j))) {
                count[s.charAt(j)-'A']++;
                countNum++;
                if (countNum >= t.length() && match(origin, count)) {
                    answer = record(i, j, s, answer);
                    while (countNum >= t.length() && match(origin, count)) {
                        if (map.containsKey(s.charAt(i))) {
                            count[s.charAt(i)-'A']--;
                            countNum--;
                        }
                        i++;
                        if (match(origin, count))
                            answer = record(i, j, s, answer);
                    }
                }
            }
            j++;
        }
        return answer;
    }

    private String record(int i, int j, String s, String answer) {
        if (s.substring(i, j+1).length() < answer.length() || answer.equals(""))
            return s.substring(i, j+1);
        else
            return answer;
    }

    private boolean match(int[] origin, int[] count) {
        for (int i = 0; i < origin.length; i++) {
            if (origin[i] > count[i])
                return false;
        }
        return true;
    }

    public String window(String s, String t) {
        int [] map = new int[128];
        for (char c : t.toCharArray()) {
            map[c]++;
        }
        int start = 0, end = 0, minStart = 0, minLen = Integer.MAX_VALUE, counter = t.length();
        while (end < s.length()) {
            final char c1 = s.charAt(end);
            if (map[c1] > 0) counter--;
            map[c1]--;
            end++;
            while (counter == 0) {
                if (minLen > end - start) {
                    minLen = end - start;
                    minStart = start;
                }
                final char c2 = s.charAt(start);
                map[c2]++;
                if (map[c2] > 0) counter++;
                start++;
            }
        }

        return minLen == Integer.MAX_VALUE ? "" : s.substring(minStart, minStart + minLen);
    }

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> answer = new ArrayList<List<Integer>>();
        for (int i = 0; i <= nums.length; i++) {
            search(answer, new ArrayList<Integer>(), i, nums, 0);
        }
        return answer;
    }

    private void search(List<List<Integer>> answer, List<Integer> temp, int l, int[] nums, int index) {
        if (l == 0) {
            answer.add(new ArrayList(temp));
        } else {
            for (int i = index; i + l < nums.length; i++) {
                temp.add(nums[i]);
                search(answer, temp, l-1, nums, i+1);
                temp.remove(temp.size()-1);
            }
        }
    }

    public int removeDuplicates(int[] nums) {
        int i = 0;
        for (int n : nums)
            if (i < 2 || n > nums[i-2])
                nums[i++] = n;
        return i;
    }

    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m + n - 1;
        while (m > 0 || n > 0) {
            if (n == 0 || nums1[m-1] >= nums2[n-1]) {
                nums1[i] = nums1[m-1];
                m--;
            } else if (m== 0 || nums1[m-1] < nums2[n-1]) {
                nums1[i] = nums2[n-1];
                n--;
            }
            i--;
        }
    }

    public int numDecodings(String s) {
        int[] dp = new int[s.length()];
        if (s.charAt(0) == '0')
            return 0;
        dp[0] = 1;
        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i) != '0')
                dp[i] = dp[i-1];
            if (i >= 1 && s.charAt(i-1) != '0' && (s.charAt(i-1) <='2' && s.charAt(i) <= '6')) {
                if (i >= 2)
                    dp[i] = dp[i] + dp[i-2];
                else
                    dp[i] = dp[i] + 1;
            }
        }
        return dp[s.length()-1];
    }

    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode cur = head;
        ListNode pre = new ListNode(0, head);
        ListNode next = head.next;
        int count = 1;
        while (count < left)
            pre = pre.next;
        cur = pre.next;
        ListNode inner = pre;
        count = left;
        while (count <= right) {
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
            count++;
        }
        inner.next.next = cur;
        inner.next = pre;
        return head;
    }

    public List<String> restoreIpAddresses(String s) {
        List<String> answer = new ArrayList<>();
        if (s.length() < 4 || s.length() > 12)
            return answer;
        backTrace(s, answer, new ArrayList<>(), 0);
        return answer;
    }

    private void backTrace(String s, List<String> answer, List<String> temp, int index) {
        if (temp.size() == 4 && index == s.length()) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 4; i++) {
                sb.append(temp.get(i));
                sb.append('.');
            }
            sb.delete(sb.length()-1, sb.length());
            answer.add(sb.toString());
        } else {
            for (int i = index; i <= s.length() && i-index <= 3; i++) {
                if (valid(s, index, i)) {
                    temp.add(s.substring(index, i));
                    backTrace(s, answer, temp, i+1);
                    temp.remove(temp.size()-1);
                }
            }
        }
    }

    private boolean valid(String s, int i, int j) {
        int val = Integer.valueOf(s.substring(i, j));
        if ((val != 0 && s.charAt(i) == '0') || val > 255)
            return false;
        return true;
    }

    public boolean isInterleave(String s1, String s2, String s3) {
        if (s1.length() + s2.length() != s3.length())
            return false;
        boolean[] match = new boolean[s1.length()+1];
        match[0] = true;
        for (int i = 0; i <= s1.length(); i++) {
            for (int j = 0; j <= s2.length(); j++) {
                if (i > 0)
                    match[j] = (s1.charAt(i-1) == s3.charAt(i+j-1) && match[j]);
                if (j > 0)
                    match[j] = (match[j] || (s2.charAt(j-1) == s3.charAt(i+j-1) && match[j-1]));
            }
        }

        Deque<Integer> queue = new LinkedList<>();

        return match[s1.length()];
    }

    public TreeNode sortedListToBST(ListNode head) {
        return build(head, null);
    }

    private TreeNode build(ListNode head, ListNode tail) {
        ListNode mid = findMid(head, tail);
        TreeNode root = new TreeNode(mid.val);
        root.left = build(head, mid);
        root.right = build(mid.next, tail);
        return root;
    }

    private ListNode findMid(ListNode head, ListNode tail) {
        ListNode fast = head;
        ListNode slow = head;
        while (fast != tail && fast.next != tail) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null)
            return false;
        if (root.left == null || root.right == null && (targetSum - root.val) == 0)
            return true;
        else {
            boolean left = (root.left == null) ? false : hasPathSum(root.left, targetSum - root.val);
            boolean right = (root.right == null) ? false : hasPathSum(root.right, targetSum - root.val);
            return left || right;
        }

    }


    public void test(TreeNode treeNode, TreeNode point) {
        if (treeNode.left != null) {
            point = treeNode;
            test(treeNode.left, point);
            System.out.println(point.val);
        }
    }

    public int numDistinct(String s, String t) {
        int[][] dp = new int[s.length()+1][t.length()+1];
        for (int i = 0; i < s.length()+1; i++)
            dp[i][0] = 1;
        for (int j = 1; j < t.length()+1; j++)
            dp[0][j] = 0;
        for (int i = 1; i < s.length()+1; i++) {
            for (int j = 1; j < t.length()+1; j++) {
                if (s.charAt(i-1) == t.charAt(j-1))
                    dp[i][j] = dp[i-1][j-1] + dp[i-1][j];
                else
                    dp[i][j] = dp[i-1][j];
            }
        }
        return dp[s.length()][t.length()];
    }

    public Node connect(Node root) {
        Deque<Node> queue = new LinkedList<>();
        if (root == null)
            return root;
        queue.addLast(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                Node node = queue.pollFirst();
                if(i != size-1)
                    node.next = queue.peekFirst();
                queue.addLast(node.left);
                queue.addLast(node.right);
            }
        }
        return root;
    }

    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> result = new LinkedList<List<Integer>>();
        LinkedList<Integer> temp = new LinkedList<Integer>();
        temp.add(1);
        result.add(new LinkedList<Integer>(temp));
        for (int i = 1; i < numRows; i++) {
            LinkedList<Integer> list = new LinkedList<Integer>();
            list.add(1);
            for (int j = 0; j < temp.size()-1; j++) {
                int a = temp.remove();
                int b = temp.peek();
                list.add(a+b);
            }
            list.add(1);
            result.add(new LinkedList<Integer>(list));
            temp = list;
        }
        return result;
    }

    public int minimumTotal(List<List<Integer>> triangle) {
        List<Integer> distance = new ArrayList<Integer>();
        int min = Integer.MAX_VALUE;
        distance.add(triangle.get(0).get(0));
        for (int i = 1; i < triangle.size(); i++) {
            List<Integer> list = triangle.get(i);
            distance.add(distance.get(distance.size()-1) + list.get(list.size()-1));
            for (int j = i-1; j > 0; j--) {
                distance.set(j, Math.min(distance.get(j), distance.get(j-1)) + list.get(j));
            }
            distance.set(0, distance.get(0) + list.get(0));
        }
        for (int i = 0; i < distance.size(); i++){
            if (min > distance.get(i))
                min = distance.get(i);
        }
        return min;
    }

    public int maxProfit(int[] prices) {
        int b1 = -prices[0], b2 = -prices[0];
        int s1 = 0, s2 = 0;
        for (int i = 1; i < prices.length; i++) {
            b1 = Math.max(b1, -prices[i]);
            s1 = Math.max(s1, b1 + prices[i]);
            b2 = Math.max(b2, s1 - prices[i]);
            s2 = Math.max(s2, b2 + prices[i]);
        }
        return s1 + s2;
    }

    public boolean isPalindrome(String s) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            if (Character.isLetterOrDigit(s.charAt(i))) {
                builder.append(Character.toLowerCase(s.charAt(i)));
            }
        }
        int left = 0, right = 0;
        if (builder.length() % 2 == 0) {
            left = builder.length() / 2  - 1;
            right = builder.length() / 2;
        } else {
            left = builder.length() / 2;
            right = builder.length() / 2;
        }
        while (left >=0 && right<builder.length()) {
            if (builder.charAt(left) != builder.charAt(right))
                return false;
            left--;
            right++;
        }
        return true;
    }

    public int longestConsecutive(int[] nums) {
        if (nums.length <=1 )
            return nums.length;
        Arrays.sort(nums);
        int result = 1;
        int temp = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] - 1 == nums[i-1]) {
                temp++;
                if (temp > result)
                    result = temp;
            } else if (nums[i] - 1 == nums[i-1]) {
                continue;
            }
            else
                temp = 1;

        }
        return result;
    }


    public void solve(char[][] board) {
        int[][] visit = new int[board.length][board[0].length]; // 0 - 'X', 1 - 'O' but may change, 2 - 'o' can't change
        for (int i = 1; i < board.length-1; i++) {
            for (int j = 1; j < board[0].length; j++) {
                if (board[i][j] == 'O') {
                    shouldChange(visit, board, i, j);
                }
            }
        }
    }

    private boolean shouldChange(int[][] visit, char[][] board, int i, int j) {
        if (i == 0 || i == board.length-1 || j == 0 || j == board[0].length-1 || visit[i][j] == 2) {
            visit[i][j] = 2;
            return false;
        }
        visit[i][j] = 1;
        boolean top = judge(visit, board, i-1, j);
        boolean bottom = judge(visit, board, i+1, j);
        boolean left = judge(visit, board, i, j-1);
        boolean right = judge(visit, board, i, j+1);
        if (top && bottom && left && right) {
            visit[i][j] = 0;
            board[i][j] = 'X';
            return true;
        } else {
            visit[i][j] = 2;
            return false;
        }
    }

    private boolean judge(int[][] visit, char[][] board, int i, int j) {
        if (board[i][j] == 'X')
            return true;
        else {
            if (i == 0 || i == board.length-1 || j == 0 || j == board[0].length-1 || visit[i][j] == 2) { // can't change
                return false;
            } else { // is o and maybe we can change
                if (visit[i][j] == 0)
                    return shouldChange(visit, board, i, j);
                else // visit[i][j] == 1
                    return true;
            }

        }
    }



    boolean[][] palindrome;
    List<List<String>> answer = new ArrayList<List<String>>();
    List<String> temp = new ArrayList<String>();
    int l = 0;

    public List<List<String>> partition(String s) {
        l = s.length();
        palindrome = new boolean[l][l];
        for (int i = 0; i < l; i++)
            Arrays.fill(palindrome[i], true);
        for (int i = l-1; i >= 0; i--) {
            for (int j = i+1; j < l; j++)
                palindrome[i][j] = palindrome[i+1][j-1] && s.charAt(i) == s.charAt(j);
        }
        dfs(s, 0);
        return answer;
    }

    private void dfs(String s, int i) {
        if (i == l)
            answer.add(new ArrayList<>(temp));
        else {
            for (int j = i; j < l; j++) {
                if (palindrome[i][j]) {
                    temp.add(s.substring(i, j+1));
                    dfs(s, j+1);
                    temp.remove(temp.size() - 1);

                }
            }
        }
    }

    public int canCompleteCircuit(int[] gas, int[] cost) {
        int[] nums = gas;
        for (int i = 0; i < gas.length; i++)
            nums[i] = gas[i] - cost[i];
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] >= 0) {
                int num = nums[i];
                int count = 1;
                while (num >= 0 && count < gas.length) {
                    num = num + nums[(i+count) % gas.length];
                    count++;
                }
                if (count == gas.length)
                    return i;
                else
                    i = i + count;
            }
        }
        return -1;
    }

    public int singleNumber(int[] nums) {
        int result = 0;
        for (int i = 0; i < 32; i++) {
            int num = 0;
            for (int j = 0; j < nums.length; j++) {
                num = num + (nums[j] & 1);
                nums[j] = nums[j] >> 1;
            }
            result = result | ((num % 3) << i);
        }
        return result;
    }


    public List<String> wordBreak(String s, List<String> wordDict) {
        HashSet<String> set = new HashSet<String>(wordDict);
        HashMap<Integer, ArrayList<String>> map = new HashMap<>();
        boolean[] dp = new boolean[s.length()+1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && set.contains(s.substring(j, i))) {
                    dp[i] = true;
                    ArrayList<String> list = map.getOrDefault(i, new ArrayList<String>());
                    ArrayList<String> prelist;
                    if (map.containsKey(j))
                        prelist = map.get(j);
                    else {
                        prelist = new ArrayList<String>();
                        prelist.add("");
                        map.put(j, prelist);
                    }
                    for (String item : prelist) {
                        StringBuilder builder = new StringBuilder(item);
                        if (builder.length() > 0)
                            builder.append(" ");
                        builder.append(s.substring(j, i));
                        list.add(builder.toString());
                    }
                    map.put(i, list);
                }
            }
        }
        return map.getOrDefault(s.length(), new ArrayList<String>());
    }

    public void reorderList(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null ) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode next =slow.next;
        slow.next = null;
        slow = next;
        ListNode sec = reverse(slow);
        merge(head, sec);
    }


    private void merge(ListNode fir, ListNode sec) {
        while (fir != null && sec != null) {
            ListNode firNext = fir.next;
            fir.next = sec;
            sec = sec.next;
            fir.next.next = firNext;
            fir = firNext;
        }
    }

    public ListNode insertionSortList(ListNode head) {
        ListNode oriPre = new ListNode(Integer.MIN_VALUE, head);
        while (head != null) {
            ListNode pre = oriPre;
            while (pre.next != head && head.val > pre.next.val) {
                pre = pre.next;
            }
            ListNode next = head.next;
            if (pre.next != head) {
                head.next = pre.next;
                pre.next = head;
                head.next.next = next;
            }
            head = next;
        }
        return oriPre.next;
    }

    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        for (String token : tokens) {
            char c = token.charAt(0);
            if (c >= '0' && c <= '9') {
                int num = Integer.parseInt(token);
                stack.push(num);
            } else {
                int num2 = stack.pop();
                int num1 = stack.pop();
                int result = 0;
                switch (c) {
                    case '+':
                        result = num1 + num2;
                        break;
                    case '-':
                        result = num1 - num2;
                        break;
                    case '*':
                        result = num1 * num2;
                        break;
                    case '/':
                        result = num1 / num2;
                }
                stack.push(result);
            }
        }
        return stack.pop();
    }

    public int findMin(int[] nums) {
        int i = 0, j = nums.length - 1;
        if (nums[i] <= nums[j])
            return nums[0];
        while (i < j) {
            int mid = (i + j) / 2;
            if (nums[mid] >= nums[0])
                i = mid + 1;
            else
                j = mid;
        }
        return nums[i];

    }


    class MinStack {

        Stack<Pair<Integer, Integer>> stack;
        int min = Integer.MIN_VALUE;
        public MinStack() {
            stack = new Stack<>();
        }

        public void push(int val) {
            if (stack.isEmpty() || min > val)
                min = val;
            stack.push(new Pair<Integer, Integer>(val, min));
        }

        public void pop() {
            stack.pop();
        }

        public int top() {
            return stack.peek().getFirst();
        }

        public int getMin() {
            return stack.peek().getSecond();
        }
    }


    public int compareVersion(String version1, String version2) {
        String[] v1 = version1.split("\\.");
        String[] v2 = version2.split("\\.");
        int i = 0;
        for (i = 0; i < Math.min(v1.length, v2.length); i++) {
            if (Integer.parseInt(v1[i]) < Integer.parseInt(v2[i]))
                return -1;
            else if (Integer.parseInt(v1[i]) > Integer.parseInt(v2[i]))
                return 1;
        }
        String[] longer;
        int possibleResult;
        if (v1.length < v2.length) {
            longer = v2;
            possibleResult = 1;
        }
        else {
            longer = v1;
            possibleResult = -1;
        }
        for (; i < longer.length; i++) {
            if (Integer.parseInt(longer[i]) != 0) {
                return possibleResult;
            }
        }
        return 0;
    }

    public int maximumGap(int[] nums) {
        if (nums.length < 2)
            return 0;
        int max = Math.max(nums[0], nums[1]);
        int min = Math.min(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            if (nums[i] > max)
                max = nums[i];
            else if (nums[i] < min)
                min = nums[i];
        }
        int gap = Math.max((max - min) / (nums.length - 1), 1);
        int[][] record = new int[(max - min) / gap + 1][2];
        for (int i = 0; i < record.length; i++)
            Arrays.fill(record[i], Integer.MIN_VALUE);
        for (int i = 0; i < nums.length; i++) {
            int id =(nums[i] - min) / gap;
            if (nums[i] > record[id][1]) {
                if (record[id][0] == Integer.MIN_VALUE)
                    record[id][0] = nums[i];
                record[id][1] = nums[i];
            }
            else if (nums[i] < record[id][0])
                record[id][0] = nums[i];
        }
        int maxGap = Integer.MIN_VALUE;
        int pre = 0;
        for (int i = 1; i < record.length; i++) {
            if (record[i][0] == Integer.MIN_VALUE)
                continue;
            else if (maxGap < (record[i][0] - record[pre][1]))
                maxGap = record[i][0] - record[pre][1];
            pre = i;
        }
        return maxGap;
    }

    public String convertToTitle(int columnNumber) {
        if (columnNumber == 1)
            return "A";
        StringBuilder builder = new StringBuilder();
        int remind = columnNumber;
        while (columnNumber > 0) {
            remind = columnNumber % 26;
            if (remind > 0)
                builder.append((char) ((remind - 1) + 'A'));
            else
                builder.append('Z');
            columnNumber = columnNumber / 26;
        }
        return builder.reverse().toString();
    }

    public int calculateMinimumHP(int[][] dungeon) {
        int m = dungeon.length;
        int n = dungeon[0].length;
        int[][] dp = new int[m][n];
        dp[m-1][n-1] = dungeon[m-1][n-1] > 0 ? 1 : -dungeon[m-1][n-1] + 1;
        for (int i = m-1; i >=0 ; i--) {
            for (int j = n-1; j >= 0; j--) {
                int num = 0;
                if (i == m-1 && j == n-1)
                    continue;
                else if (i == m-1)
                    num = dp[i][j+1];
                else if (j == n-1)
                    num = dp[i+1][j];
                else
                    num = Math.min(dp[i][j+1], dp[i+1][j]);
                if (dungeon[i][j] > 0)
                    num = dungeon[i][j] - num >= 0 ? 1 : num - dungeon[i][j];
                else
                    num = num - dungeon[i][j];
                dp[i][j] = num;
            }
        }
        return dp[0][0];
    }

    public String largestNumber(int[] nums) {
        String[] strings = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            strings[i] = Integer.toString(nums[i]);
        }
        Comparator<String> comparator = new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                String s1 = o1 + o2;
                String s2 = o2 + o1;
                return s2.compareTo(s1);
            }
        };
        Arrays.sort(strings, comparator);
        if (strings[0].charAt(0) == '0')
            return "0";
        StringBuilder builder = new StringBuilder();
        for (String s : strings) {
            builder.append(s);
        }
        return builder.toString();
    }

    public int reverseBits(int n) {
        int rev = 0;
        for (int i = 0; i < 32 && n != 0; ++i) {
            rev |= (n & 1) << (31 - i);
            n >>>= 1;
        }
        return rev;
    }
    public List<Integer> rightSideView(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        List<Integer> result = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                if (node.left != null)
                    queue.add(node.left);
                if (node.right != null)
                    queue.add(node.right);
                if (i == size - 1)
                    result.add(node.val);
            }
        }
        return result;
    }

    public int rangeBitwiseAnd(int left, int right) {
        if (left == right)
            return left;
        int size = (right - left);
        int bits = 0;
        while (size != 0) {
            size = size >> 1;
            bits++;
        }
        int flag = Integer.MAX_VALUE << bits;
        int answer = flag & right;
        return answer;
    }

    public boolean isIsomorphic(String s, String t) {
        return (convert(s).equals(convert(t)));
    }

    private String convert(String s) {
        StringBuilder sb = new StringBuilder();
        HashMap<Character, String> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))) {
                sb.append(map.get(s.charAt(i)));
            } else {
                String val = map.size() < 10 ?  "0" + map.size() : Integer.toString(map.size());
                map.put(s.charAt(i), val);
                sb.append(val);
            }
        }
        return sb.toString();
    }

    public void mapArrayTest() {
        HashMap<Integer, List<Integer>> map = new HashMap<>();
        map.put(1, new ArrayList<>());
        map.get(1).add(1);
        List<Integer> list = map.get(1);
        list.add(2);
        map.put(1, list);
        list = map.get(1);
        for (Integer i : list) {
            System.out.println(i + "???");
        }
    }

    public void charArrayTest() {
        char[] array = new char[10];
    }

    public void intArrayTest() {
        int[] array = new int[10];
        array[2]++;
        array[3] = array[3] + 1;
        for (int i : array)
            System.out.println(i);
    }

    public int minSubArrayLen(int target, int[] nums) {
        int fast = 0, slow = 0;
        int sum = 0, length = Integer.MAX_VALUE;
        while (slow != nums.length) {
            while (sum < target && fast < nums.length) {
                sum = sum + nums[fast];
                fast++;
            }
            if (sum >= target && length > (fast - slow))
                length = fast - slow + 1;
            sum = sum - nums[slow];
            slow++;
            if (length == Integer.MAX_VALUE && fast == nums.length)
                break;
        }
        return length == Integer.MAX_VALUE ? 0 : length;
    }

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] degree = new int[numCourses];
        HashMap<Integer, List<Integer>> pres = new HashMap<>();
        int[] result = new int[numCourses];
        int index = 0;
        for (int i = 0; i < prerequisites.length; i++) {
            degree[prerequisites[i][0]]++;
            List<Integer> list = pres.getOrDefault(prerequisites[i][1], new ArrayList<Integer>());
            list.add(prerequisites[i][0]);
            pres.put(prerequisites[i][1], list);
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (degree[i] == 0)
                queue.add(i);
        }
        while (!queue.isEmpty()) {
            int i = queue.poll();
            result[index] = i;
            index++;
            if (pres.containsKey(i)) {
                List<Integer> list = pres.get(i);
                for (int num : list) {
                    degree[num]--;
                    if (degree[num] == 0)
                        queue.add(num);
                }
            }
        }
        if (index == numCourses)
            return result;
        else
            return new int[0];
    }

    public void stringBuilderTest() {
        StringBuilder builder = new StringBuilder();
        builder.reverse();

    }

    public void wordDictionaryInit() {
        WordDictionary wordDictionary = new WordDictionary();
        wordDictionary.addWord("bad");
        wordDictionary.search("b..");
    }

    class WordDictionary {

        class Trie {
            Trie[] children = new Trie[26];
        }

        Trie root;

        public WordDictionary() {
            root = new Trie();
        }

        public void addWord(String word) {
            Trie cur = root;
            for (int i = 0; i < word.length(); i++) {
                if (cur.children[word.charAt(i) - 'a'] == null) {
                    cur.children[word.charAt(i) - 'a'] = new Trie();
                }
                cur = cur.children[word.charAt(i) - 'a'];
            }
        }

        public boolean search(String word) {
            return partSearch(word, 0, root);
        }

        private boolean partSearch(String word, int index, Trie root) {
            Trie cur = root;
            for (int i = index; i < word.length(); i++) {
                if (word.charAt(i) == '.') {
                    boolean find = false;
                    for (int j = 0; j < 26; j++) {
                        if (cur.children[j] != null)
                            find = find || partSearch(word, i+1, cur.children[j]);
                    }
                    return find;
                } else {
                    if (cur.children[word.charAt(i) - 'a'] == null)
                        return false;
                }
                cur = cur.children[word.charAt(i) - 'a'];
            }
            return true;

        }
    }

    class Trie {
        Trie[] children = new Trie[26];
        boolean isEnd = false;
    }

    Trie root;
    public List<String> findWords(char[][] board, String[] words) {
        root = new Trie();
        int maxLength = 0;
        HashSet<String> set = new HashSet<>();
        List<String> list = new ArrayList<>();
        for (String s : words) {
            Trie cur = root;
            for (int i = 0; i < s.length(); i++) {
                if (cur.children[s.charAt(i) - 'a'] == null)
                    cur.children[s.charAt(i) - 'a'] = new Trie();
                cur = cur.children[s.charAt(i) - 'a'];
            }
            cur.isEnd = true;
            if (s.length() > maxLength)
                maxLength = s.length();
        }

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                StringBuilder sb = new StringBuilder();
                if (prefix(sb.toString()))
                    dfs(board, sb, i, j, set, maxLength);
            }
        }
        for (String s : set) {
            list.add(s);
        }
        return list;
    }

    private void dfs (char[][] board, StringBuilder sb, int i, int j, HashSet<String> set, int maxLength) {
        if (validOnTrie(sb.toString())) {
            set.add(sb.toString());
        }
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] == '!')
            return;
        if (sb.length() <= maxLength) {
            if (prefix(sb.toString())) {
                sb.append(board[i][j]);
                board[i][j] = '!';
                int length = sb.length();
                dfs(board, sb, i+1, j, set, maxLength);
                sb.delete(length, sb.length());
                dfs(board, sb, i-1, j, set, maxLength);
                sb.delete(length, sb.length());
                dfs(board, sb, i, j+1, set, maxLength);
                sb.delete(length, sb.length());
                dfs(board, sb, i, j-1, set, maxLength);
                sb.delete(length, sb.length());
                board[i][j] = sb.charAt(sb.length()-1);

            }
        }
    }

    private boolean validOnTrie(String s) {
        Trie trie = search(s);
        return trie == null  ? false : trie.isEnd;
    }

    private boolean prefix(String s) {
        Trie trie = search(s);
        return (trie != null);
    }

    private Trie search(String s) {
        Trie cur = root;
        for (int i = 0; i < s.length(); i++) {
            if (cur.children[s.charAt(i) - 'a'] == null)
                return null;
            else
                cur = cur.children[s.charAt(i) - 'a'];
        }
        return cur;
    }

    public String shortestPalindrome(String s) {
        int length = s.length();
        boolean self = (length % 2 == 1);
        HashMap<Integer, Character> map = new HashMap<>();
        for (int i = 0; i < length; i++)
            map.put(i, s.charAt(i));
        for (int i = length / 2; i >= 0; i--) {
            if (isPalindrome(i, s, self, map))
                return count(i, length, self, s);
            else {
                self = !self;
                if (!self) {
                    if (isPalindrome(i, s, self, map))
                        return count(i, length, self, s);
                }
            }
            self = true;
        }
        return "";
    }

    private String count(int pivot, int length, boolean self, String s) {
        StringBuilder builder;
        if (self) {
            builder = new StringBuilder(s.substring(2*pivot+1, length));
            builder.reverse();
            builder.append(s.substring(0, length));
        }
        else {
            builder = new StringBuilder(s.substring(2*pivot, length));
            builder.reverse();
            builder.append(s.substring(0, length));
        }
        return builder.toString();
    }
    // length-1-i - i
    // length-i--i

    private boolean isPalindrome(int pivot, String s, boolean self, HashMap<Integer, Character> map) {
        int l = pivot - 1;
        int r = pivot + 1;
        if (!self) {
            l = pivot - 1;
            r = pivot;
        }
        while (l >= 0 && r < s.length() && map.get(l) == map.get(r)) {
            l--;
            r++;
        }
        if (l == -1)
            return true;
        else
            return false;
    }

    int result = 0;
    public int findKthLargest(int[] nums, int k) {
        return quickSort(nums, 0, nums.length - 1, k-1);
    }

    private int quickSort(int[] nums, int l, int r, int k) {
        int mid = (l + r) / 2;
        int temp = nums[mid];
        int i = l+1, j = r;
        swap(nums, mid, l);
        while (i < j) {
            while (nums[j] <= temp && j > i)
                j--;
            while (nums[i] >= temp && j > i)
                i++;

            if (i < j)
                swap(nums, i, j);
        }
        nums[l] = nums[i];
        nums[i] = temp;
        if (i == k)
            return nums[i];
        else if (i < k)
            return quickSort(nums, i + 1, r, k);
        else
            return quickSort(nums, l, i - 1, k);
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public void testStack() {
        MyStack stack = new MyStack();
        stack.push(1);
        stack.push(2);
        stack.push(3);
        stack.pop();
        stack.pop();
        stack.pop();

    }

    class MyStack {

        Queue<Integer> queue1;
        Queue<Integer> queue2;
        int top;
        public MyStack() {
            queue1 = new LinkedList<>();
            queue2 = new LinkedList<>();
        }

        public void push(int x) {
            queue1.add(x);
            top = x;
        }

        public int pop() {
            for (int i = 0; i < queue1.size()-1; i++) {
                queue2.add(queue1.poll());
            }
            int result = queue1.poll();
            for (int i = 0; i < queue2.size()-1; i++) {
                queue1.add(queue2.poll());
            }
            if (!queue2.isEmpty()) {
                top = queue2.poll();
                queue1.add(top);
            }
            return result;
        }

        public int top() {
            return top;
        }

        public boolean empty() {
            return queue1.isEmpty();
        }
    }

    public int calculate(String s) {
        Deque<Integer> stack = new ArrayDeque<Integer>();
        char preSign = '+';
        int num = 0;
        int n = s.length();
        for (int i = 0; i < n; ++i) {
            if (Character.isDigit(s.charAt(i))) {
                num = num * 10 + s.charAt(i) - '0';
            }
            if (!Character.isDigit(s.charAt(i)) && s.charAt(i) != ' ' || i == n - 1) {
                switch (preSign) {
                    case '+':
                        stack.push(num);
                        break;
                    case '-':
                        stack.push(-num);
                        break;
                    case '*':
                        stack.push(stack.pop() * num);
                        break;
                    default:
                        stack.push(stack.pop() / num);
                }
                preSign = s.charAt(i);
                num = 0;
            }
        }
        int ans = 0;
        while (!stack.isEmpty()) {
            ans += stack.pop();
            StringBuilder stringBuilder = new StringBuilder();
        }
        return ans;
    }

    public boolean isPowerOfTwo(int n) {
        while (n/2 > 1 && n % 2 == 0)
            n = n / 2;
        return n == 1;
    }

    public String encry(String c1, String p1, String p2) {

        int key[] = new int[c1.length()];
        for (int i = 0; i < p1.length(); i++) {
            key[i] = (c1.charAt(i) - 'A' + 26 - (p1.charAt(i) - 'A')) % 26;
        }
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < p2.length(); i++) {
            stringBuilder.append((char) ((p2.charAt(i) - 'A' + key[i])%26 + 'A'));
        }
        System.out.println(stringBuilder.toString());
        return stringBuilder.toString();
    }

    public boolean isPalindrome(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        ListNode pre = null;
        if (head.next == null)
            return true;
        while (fast != null && fast.next != null) {
            pre = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode reverse = reverse(slow);
        if (fast == null) {  // even
            pre.next = null;
        } else {
            pre.next = new ListNode(pre.next.val);
        }
        return compare(head, reverse);
    }

    private ListNode reverse(ListNode node) {
        ListNode pre = null;
        ListNode next = node.next;
        while (node != null) {
            next = node.next;
            node.next = pre;
            pre = node;
            node = next;
        }
        return pre;
    }

    private boolean compare(ListNode n1, ListNode n2) {
        if (n1 == null && n2 == null)
            return true;
        else if (n1 == null || n2 == null)
            return false;
        else {
            if (n1.val != n2.val)
                return false;
        }
        return true;
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        TreeNode left = p.val < q.val ? p : q;
        TreeNode right = p.val > q.val ? p : q;
        TreeNode result = null;
        while (result == null) {
            if (root.val < left.val)
                root = root.right;
            else if (root.val > right.val)
                root = root.left;
            else
                result = root;
        }
        return root;
    }

    public int timeRequiredToBuy(int[] tickets, int k) {
        int count = 0;
        for (int i = 0; i < tickets.length; i++) {
            if (i <= k)
                count = count + (tickets[k] > tickets[i]?tickets[i] : tickets[k]);
            else
                count = count + (tickets[k] > tickets[i]?tickets[i] : tickets[k]-1);
        }
        return count;
    }

    public int minimunCost(int n, int[][] hightways, int disconts) {
        List<int[]>[] g = new List[n];
        for (int i = 0; i < n; i++) {
            g[i] = new ArrayList<>();
        }
        for (int i =0; i < hightways.length; i++) {
            int a = hightways[i][0];
            int b = hightways[i][1];
            int c = hightways[i][2];
            g[a].add(new int[]{b, c});
            g[b].add(new int[]{a, c});
        }
        PriorityQueue<int[]> queue = new PriorityQueue<>((a, b) -> a[0] - b[0]);
        int[][] distance = new int[n][disconts+1];
        for (int i = 0; i < distance.length; i++)
            Arrays.fill(distance[i], Integer.MAX_VALUE);
        queue.add(new int[]{0,0,0}); //  cost,destination, discount;
        while (!queue.isEmpty()) {
            int[] temp = queue.poll();
            int cost = temp[0], d = temp[1], dis = temp[2];
            if (dis > disconts || cost > distance[d][dis])
                continue;
            distance[d][dis] = cost;
            if (d == n-1)
                return cost;
            for (int[] dest : g[d]) {
                queue.add(new int[]{dest[1] + cost, dest[0],  dis});
                queue.add(new int[]{dest[1] / 2 + cost, dest[0], dis+1});
            }
        }
        return distance[n-1][disconts];
    }

    public void generateKey(String x, String y, String z) {
        StringBuilder xb = new StringBuilder(x);
        StringBuilder yb = new StringBuilder(y);
        StringBuilder zb = new StringBuilder(z);
        StringBuilder f = new StringBuilder();
        int length = 64;
        for (int i = 0; i < length; i++) {
            xb.append((xb.charAt(i) - '0') ^ (xb.charAt(i+1) - '0'));
            yb.append((yb.charAt(i) - '0') ^ (yb.charAt(i+3) - '0'));
            zb.append((zb.charAt(i) - '0') ^ (zb.charAt(i+2) - '0'));
            f.append(((xb.charAt(i) - '0') & (yb.charAt(i) - '0')) ^ (zb.charAt(i) - '0'));
        }
        System.out.println("x: " + xb.substring(0, length).toString());
        System.out.println("y: " + yb.substring(0, length).toString());
        System.out.println("z: " + zb.substring(0, length).toString());
        System.out.println("f: " + f.toString());
    }

    int arrayCount(int[] a, int m, int k) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int count = 0;
        boolean addFirst = false;
        int valid = -m;
        for (int r = 0; r < a.length; r++) {
            if (map.containsKey(k - a[r])) {
                if (map.get(k-a[r]) + m > r) {
                    if (r < m && !addFirst) {
                        count++;
                        addFirst = true;
                    }
                    else if (r >= m)
                        count++;
                    valid = map.get(k-a[r]);
                } else {
                    if (valid + m > r)
                        count++;
                }
            } else {
                if (valid + m > r)
                    count++;
            }
            map.put(a[r], r);
        }
        return count;
    }

    public List<Integer> memory(String[][] requests, int totalSlots) {
        boolean[] memo = new boolean[totalSlots];
        List<Integer> result = new LinkedList<>();
        for (int i = 0; i < requests.length; i++) {
            int start = Integer.parseInt(requests[i][1]);
            int length = Integer.parseInt(requests[i][2]);
            if ("store".equals(requests[i][0])) {
                boolean add = false;
                for (int count = 0; count < totalSlots; count++) {
                    boolean find = true;
                    for (int j = start; j - start < length; j++) {
                        if (memo[j%totalSlots]) {
                            count = count + j - start;
                            start = (j+1)%totalSlots;
                            find = false;
                            break;
                        }
                    }
                    if (find) {
                        for (int j = start; j - start< length; j++)
                            memo[j%totalSlots] = true;
                        result.add(start);
                        add = true;
                        break;
                    }
                }
                if (!add)
                        result.add(-1);
            } else {
                for (int j = 0; j < length; j++) {
                    memo[(start+j)%totalSlots] = false;
                }
                result.add(length);
            }
        }
        return result;
    }

    public List<Integer> diffWaysToCompute(String expression) {
        return process(expression, 0, expression.length()-1);
    }

    private List<Integer> process(String ex, int l, int r) {
        List<Integer> list = new LinkedList<>();
        for (int i = 0; i <= r; i++) {
            char c = ex.charAt(i);
            if (!Character.isDigit(c)) {
                List<Integer> left = process(ex, l, i-1);
                List<Integer> right = process(ex, i+1, r);
                for (int a : left) {
                    for (int b: right) {
                        if (c == '+')
                            list.add(a+b);
                        else if (c == '-')
                            list.add(a-b);
                        else
                            list.add(a*b);
                    }
                }
            }
        }
        if (list.size() == 0) { //from l to r is a number
            int num = 0;
            for (int i = l; i <= r; i++)
                num = num *10 + ex.charAt(i) - '0';
            list.add(num);
        }
        return list;
    }

    public int[] singleNumber2(int[] nums) {
        int xor = 0;
        for ( int num : nums) {
            xor = xor ^ num;
        }
        int lastBit = 1;
        int count = 0;
        while (xor % 2 == 0) {
            xor = xor >> 1;
            lastBit = lastBit << 1;
            count++;
        }
        int type1 = 0, type2 = 0;
        for (int num: nums) {
            if ((((lastBit ^ num) >> count) & 1) == 0)
                type1 = type1 ^ num;
            else
                type2 = type2 ^ num;
        }
        return new int[]{type1, type2};
    }
}

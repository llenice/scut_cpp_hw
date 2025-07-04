#include <iostream>   
#include <string>     
#include <vector>     
#include <algorithm>  
#include <stdexcept>  
#include <sstream>
using namespace std;

// 每个节点存储的位数（基数为10^BASE_DIGITS）
const int BASE_DIGITS = 4;  // 每个节点存储4位数字
const int BASE = 10000;     // 基数为10000

struct ListNode {
    int data;           // 存储的数字（0 <= data < BASE）
    ListNode* next;     // 指向下一个节点的指针
    
    ListNode(int val = 0) : data(val), next(nullptr) {}
};


class BigNumber {
private:
    ListNode* head;           // 链表头指针（指向最低位）
    bool is_negative;         // 是否为负数
    int decimal_position;     // 小数点位置（从右数第几位）
    string error_message;     // 错误信息

public:
    /**
     * 构造函数
     */
    BigNumber(const string& s = "0") : head(nullptr), is_negative(false), decimal_position(0) {
        if (s == "ERROR") {
            error_message = "ERROR";
            return;
        }
        parseFromString(s);
    }

    /**
     * 拷贝构造函数
     */
    BigNumber(const BigNumber& other) : head(nullptr), is_negative(other.is_negative), 
                                       decimal_position(other.decimal_position), 
                                       error_message(other.error_message) {
        copyFrom(other);
    }

    ~BigNumber() {
        clear();
    }

    /**
     * 赋值操作符
     */
    BigNumber& operator=(const BigNumber& other) {
        if (this != &other) {
            clear();
            is_negative = other.is_negative;
            decimal_position = other.decimal_position;
            error_message = other.error_message;
            copyFrom(other);
        }
        return *this;
    }

    /**
     * 清空链表
     */
    void clear() {
        while (head) {
            ListNode* temp = head;
            head = head->next;
            delete temp;
        }
        head = nullptr;
    }

    /**
     * 复制数据
     */
    void copyFrom(const BigNumber& other) {
        if (!other.head) {
            head = new ListNode(0);
            return;
        }
        
        ListNode* otherCur = other.head;
        head = new ListNode(otherCur->data);
        ListNode* cur = head;
        otherCur = otherCur->next;
        
        while (otherCur) {
            cur->next = new ListNode(otherCur->data);
            cur = cur->next;
            otherCur = otherCur->next;
        }
    }

    /**
     * 从带有符号的字符串解析数字，构造BigNum类实例
     */
    void parseFromString(const string& str) {
        clear();
        string s = str;
        // 移除所有逗号
        s.erase(remove(s.begin(), s.end(), ','), s.end());
        
        if (s.empty() || s == ".") s = "0";    

        // 处理负号
        if (!s.empty() && s[0] == '-') {
            is_negative = true;
            s = s.substr(1);
        }
        
        // 查找小数点
        size_t dot_pos = s.find('.');
        if (dot_pos == string::npos) {
            // 整数
            decimal_position = 0;
            createList(s);
        } else {
            // 有小数部分
            string int_part = s.substr(0, dot_pos);
            string frac_part = s.substr(dot_pos + 1);
            
            if (int_part.empty()) int_part = "0";
            
            decimal_position = frac_part.length();
            createList(int_part + frac_part);  // 合并为纯数字
        }
        
    }

    /**
     * 从纯数字字符串创建链表
     */
    void createList(const string& s) {
        string digits = s;
        //创建链表节点
        head = nullptr;
        ListNode* tail = nullptr;
        
        int len = digits.length();
        for (int i = len; i > 0; i -= BASE_DIGITS) {
            int start = max(0, i - BASE_DIGITS);
            string segment = digits.substr(start, i - start);
            int value = stoi(segment);
            
            ListNode* newNode = new ListNode(value);
            if (!head) {
                head = tail = newNode;
            } else {
                tail->next = newNode;
                tail = newNode;
            }
        }
        
        if (!head) {
            head = new ListNode(0);
        }
    }

    /**
     * 判断是否为零
     */
    bool isZero() const {
        ListNode* cur = head;
        while (cur) {
            if (cur->data != 0) return false;
            cur = cur->next;
        }
        return true;
    }

    /**
     * 从链表中移除前导零
     */
    void removeLeadingZeros() {
        if (!head) return;
        
        vector<int> digits;
        ListNode* cur = head;
        while (cur) {
            digits.push_back(cur->data);
            cur = cur->next;
        }
        
        while (digits.size() > 1 && digits.back() == 0) {
            digits.pop_back();
        }
        
        clear();
        ListNode* tail = nullptr;
        
        for (int digit : digits) {
            ListNode* newNode = new ListNode(digit);
            if (!head) {
                head = tail = newNode;
            } else {
                tail->next = newNode;
                tail = newNode;
            }
        }
        
        if (isZero()) {
            is_negative = false;
            decimal_position = 0;
        }
    }

    /**
     * 将链表转换为带符号的字符串
     * @param with_commas 是否添加千位分隔符，默认为true
     * @return 格式化的字符串表示
     */
    string toString(bool with_commas = true) const {
        // 处理错误情况
        if (!error_message.empty()) return error_message;
        if (!head || isZero()) return "0";
        
        string full_number = toIntegerString();

        // 插入小数点
        string result = "";
        if (is_negative) result += "-";
        
        string int_part, frac_part;
        
        if (decimal_position == 0) {
            // 整数
            int_part = full_number;
            frac_part = "";
        } else if (decimal_position >= full_number.length()) {
            // 纯小数
            int_part = "0";
            string zeros(decimal_position - full_number.length(), '0');
            frac_part = zeros + full_number;

        } else {
            // 有整数部分和小数部分
            int int_len = full_number.length() - decimal_position;
            int_part = full_number.substr(0, int_len);
            frac_part = full_number.substr(int_len);
        }
        
        // 移除小数部分末尾的零
        while (!frac_part.empty() && frac_part.back() == '0') {
            frac_part.pop_back();
        }
        
        // 对所有整数部分都添加千位分隔符
        if (with_commas) {
            string formatted_int = "";
            int count = 0;
            
            for (int i = int_part.length() - 1; i >= 0; i--) {
                formatted_int += int_part[i];
                count++;
                if (count == 3 && i != 0) {
                    formatted_int += ',';
                    count = 0;
                }
            }
            
            reverse(formatted_int.begin(), formatted_int.end());
            result += formatted_int;
        } else {
            result += int_part;
        }
        
        // 添加小数部分
        if (!frac_part.empty()) {
            result += "." + frac_part;
        }
        
        return result;
    }

    /**
     * 对齐两个数的小数位数
     */
    static pair<BigNumber, BigNumber> alignDecimals(const BigNumber& a, const BigNumber& b) {
        int max_decimal = max(a.decimal_position, b.decimal_position);
        
        // 获取原始字符串表示并补零对齐
        string str_a = a.toString(false);  // 不带千位分隔符
        string str_b = b.toString(false);
        
        // 移除负号用于处理
        bool neg_a = false, neg_b = false;
        if (!str_a.empty() && str_a[0] == '-') {
            neg_a = true;
            str_a = str_a.substr(1);
        }
        if (!str_b.empty() && str_b[0] == '-') {
            neg_b = true;
            str_b = str_b.substr(1);
        }
        
        // 为字符串补零对齐小数位数
        size_t dot_pos_a = str_a.find('.');
        size_t dot_pos_b = str_b.find('.');
        
        if (dot_pos_a == string::npos) str_a += ".";
        if (dot_pos_b == string::npos) str_b += ".";
        
        // 补零到相同的小数位数
        while (str_a.length() - str_a.find('.') - 1 < max_decimal) {
            str_a += "0";
        }
        while (str_b.length() - str_b.find('.') - 1 < max_decimal) {
            str_b += "0";
        }
        
        // 重新添加负号
        if (neg_a) str_a = "-" + str_a;
        if (neg_b) str_b = "-" + str_b;
        
        // 重新创建BigNumber对象
        BigNumber a_aligned(str_a);
        BigNumber b_aligned(str_b);
        
        return make_pair(a_aligned, b_aligned);
    }

    /**
     * 比较无符号的绝对值大小
     */
    int compareAbsWith(const BigNumber& other) const {
        string this_full_num = this->toIntegerString();
        string other_full_num = other.toIntegerString();
        int len_a = this_full_num.length();
        int len_b = other_full_num.length();
        if (len_a > len_b) return 1;
        if (len_a < len_b) return -1;
        if(this_full_num > other_full_num) return 1;
        else if(this_full_num < other_full_num) return -1;
        return 0;
        
    }

    /**
     * 加法运算
     */
    BigNumber operator+(const BigNumber& other) const { 
        // 符号相同，执行加法
        if (is_negative == other.is_negative) {
            auto aligned = alignDecimals(*this, other);
            BigNumber result = addAbsolute(aligned.first, aligned.second);
            result.is_negative = is_negative;
            result.decimal_position = aligned.first.decimal_position;
            return result;
        } else {
            // 符号不同，转换为减法
            BigNumber other_copy = other;
            other_copy.is_negative = !other_copy.is_negative;
            return *this - other_copy;
        }
    }

    /**
     * 减法运算
     */
    BigNumber operator-(const BigNumber& other) const {
         // 符号不同，转换为加法
        if (is_negative != other.is_negative) {
            BigNumber other_copy = other;
            other_copy.is_negative = !other_copy.is_negative;
            return *this + other_copy;
        }
        
        // 符号相同，比较绝对值
        auto aligned = alignDecimals(*this, other);
        int cmp = aligned.first.compareAbsWith(aligned.second);
        
        if (cmp == 0) {
            return BigNumber("0");
        }
        
        BigNumber result;
        if (cmp > 0) {
            result = subtractAbsolute(aligned.first, aligned.second);
            result.is_negative = is_negative;
        } else {
            result = subtractAbsolute(aligned.second, aligned.first);
            result.is_negative = !is_negative;
        }
        
        result.decimal_position = aligned.first.decimal_position;
        return result;
    }

    /**
     * 乘法运算
     */
    BigNumber operator*(const BigNumber& other) const {
        if (isZero() || other.isZero()) {
            return BigNumber("0");
        }
        
        BigNumber result = multiplyAbsolute(*this, other);
        result.is_negative = is_negative ^ other.is_negative;
        result.decimal_position = decimal_position + other.decimal_position;  
        return result;
    }

    /**
     * 除法运算
     */
    BigNumber operator/(const BigNumber& other) const {
        if (other.isZero()) return BigNumber("ERROR");

        if (other.isAbsLessThan1e_6()) {
            return BigNumber("ERROR");
        }
        
        if (isZero()) return BigNumber("0");
        
        bool result_negative = is_negative ^ other.is_negative;
        
        // 创建被除数和除数的绝对值副本
        BigNumber dividend = *this;
        BigNumber divisor = other;
        dividend.is_negative = false;
        divisor.is_negative = false;
        
        // 对齐小数位数
        auto aligned = alignDecimals(dividend, divisor);
        
        // 转换为整数字符串
        string dividend_str = aligned.first.toIntegerString();
        string divisor_str = aligned.second.toIntegerString();
        
        // 添加10位精度
        for (int i = 0; i < 10; i++) {
            dividend_str += "0";
        }
        
        // 执行整数除法
        BigNumber dividend_int(dividend_str);
        BigNumber divisor_int(divisor_str);
        
        BigNumber result = divideAbsolute(dividend_int, divisor_int);
        result.decimal_position = 10;  // 10位小数
        result.is_negative = result_negative;
    
        
        return result;
    }

    /**
     * 判断绝对值是否小于1e-6
     */
    bool isAbsLessThan1e_6() const {
        if (isZero()) return true;
        
        // 获取完整的数字字符串表示
        string str = toString(false);  // 获取不带千位分隔符的字符串
        if (!str.empty() && str[0] == '-') {
            str = str.substr(1);  // 移除负号，只考虑绝对值
        }
        
        // 查找小数点
        size_t dot_pos = str.find('.');
        
        if (dot_pos == string::npos) {
            // 纯整数
            return str == "0";
        }
        
        // 有小数部分
        string int_part = str.substr(0, dot_pos);
        string frac_part = str.substr(dot_pos + 1);
        
        // 如果整数部分不为0，则绝对值 >= 1，不可能小于1e-6
        if (int_part != "0") {
            return false;
        }
        
        // 修复：防止数组越界
        if (frac_part.length() < 6) {
            return true;  // 小数位数不足6位，肯定小于1e-6
        }
        
        // 检查前6位是否都是0
        for (int i = 0; i < 6; i++) {
            if (frac_part[i] != '0') {
                return false;
            }
        }
        
        // 如果前6位都是0，则小于1e-6
        return true;
    }

private:
    /**
     * 转换为纯整数字符串（移除小数点）
     */
    string toIntegerString() const {
        vector<int> digits;
        ListNode* cur = head;
        while (cur) {
            digits.push_back(cur->data);
            cur = cur->next;
        }
        
        string result = "";
        bool first = true;
        for (int i = digits.size() - 1; i >= 0; i--) {
            if (first) {
                result += to_string(digits[i]);
                first = false;
            } else {
                string segment = to_string(digits[i]);
                while (segment.length() < BASE_DIGITS) {
                    segment = "0" + segment;
                }
                result += segment;
            }
        }
        
        return result.empty() ? "0" : result;
    }

    /**
     * 绝对值加法
     */
    BigNumber addAbsolute(const BigNumber& a, const BigNumber& b) const {
        BigNumber result;
        result.clear();
        
        ListNode* ptr1 = a.head;
        ListNode* ptr2 = b.head;
        ListNode* tail = nullptr;
        int carry = 0;
        
        while (ptr1 || ptr2 || carry) {
            int sum = carry;
            if (ptr1) {
                sum += ptr1->data;
                ptr1 = ptr1->next;
            }
            if (ptr2) {
                sum += ptr2->data;
                ptr2 = ptr2->next;
            }
            
            ListNode* newNode = new ListNode(sum % BASE);
            carry = sum / BASE;
            
            if (!result.head) {
                result.head = tail = newNode;
            } else {
                tail->next = newNode;
                tail = newNode;
            }
        }
        
        return result;
    }

    /**
     * 绝对值减法（假设 a >= b）
     */
    BigNumber subtractAbsolute(const BigNumber& a, const BigNumber& b) const {
        BigNumber result;
        result.clear();
        
        ListNode* ptr1 = a.head;
        ListNode* ptr2 = b.head;
        ListNode* tail = nullptr;
        int borrow = 0;
        
        while (ptr1) {
            int diff = ptr1->data - borrow;
            if (ptr2) {
                diff -= ptr2->data;
                ptr2 = ptr2->next;
            }
            
            if (diff < 0) {
                diff += BASE;
                borrow = 1;
            } else {
                borrow = 0;
            }
            
            ListNode* newNode = new ListNode(diff);
            
            if (!result.head) {
                result.head = tail = newNode;
            } else {
                tail->next = newNode;
                tail = newNode;
            }
            
            ptr1 = ptr1->next;
        }
        
        result.removeLeadingZeros();
        return result;
    }

    /**
     * 绝对值乘法
     */
    BigNumber multiplyAbsolute(const BigNumber& a, const BigNumber& b) const {
        if (a.isZero() || b.isZero()) {
            return BigNumber("0");
        }
        
        vector<int> digits_a, digits_b;
        ListNode* cur = a.head;
        while (cur) { digits_a.push_back(cur->data); cur = cur->next; }
        cur = b.head;
        while (cur) { digits_b.push_back(cur->data); cur = cur->next; }
        
        vector<long long> product(digits_a.size() + digits_b.size(), 0);
        
        for (size_t i = 0; i < digits_a.size(); i++) {
            for (size_t j = 0; j < digits_b.size(); j++) {
                product[i + j] += (long long)digits_a[i] * digits_b[j];
            }
        }
        
        long long carry = 0;
        for (size_t i = 0; i < product.size(); i++) {
            product[i] += carry;
            carry = product[i] / BASE;
            product[i] %= BASE;
        }
        
        while (carry > 0) {
            product.push_back(carry % BASE);
            carry /= BASE;
        }
        
        BigNumber result;
        result.clear();
        ListNode* tail = nullptr;
        
        for (size_t i = 0; i < product.size(); i++) {
            ListNode* newNode = new ListNode(product[i]);
            if (!result.head) {
                result.head = tail = newNode;
            } else {
                tail->next = newNode;
                tail = newNode;
            }
        }
        
        result.removeLeadingZeros();
        return result;
    }

    /**
     * 模拟长除法过程
     * 1. 从被除数的最高位开始，逐位构建当前段
     * 2. 判断当前段是否能被除数整除
     * 3. 计算商的当前位，更新余数
     * 4. 继续处理下一位，直到所有位处理完毕
     */
    BigNumber divideAbsolute(const BigNumber& dividend, const BigNumber& divisor) const {
        if (divisor.isZero()) {
            throw runtime_error("Division by zero");
        }
        
        if (dividend.isZero()) {
            return BigNumber("0");
        }
        
        string dividend_str = dividend.toIntegerString();
        string divisor_str = divisor.toIntegerString();
        
        string quotient = ""; //商
        string current_segment = ""; //当前处理的数字段
        
        // 模拟长除法过程 - 从左到右逐位处理
        for (char digit : dividend_str) {
            current_segment += digit;

            // 移除当前段的前导零（但保留单个'0'）
            size_t first_digit = current_segment.find_first_not_of('0');
            if (first_digit == string::npos) {
                current_segment = "0";
            } else {
                current_segment = current_segment.substr(first_digit);
            }
            
            //  将当前段和除数转换为BigNumber进行比较
            BigNumber temp_segment(current_segment);
            BigNumber temp_divisor(divisor_str);
            temp_segment.decimal_position = 0;
            temp_divisor.decimal_position = 0;

            // 判断当前段是否能被除数整除
            if (temp_segment.compareAbsWith(temp_divisor) < 0) {
                // 当前段小于除数，商的当前位为0
                if (!quotient.empty()) {
                    quotient += "0";// 只有在商不为空时才添加0
                }
            } else {
                // 当前段大于等于除数，计算能整除几次
                int count = 0;

                // 计算商的当前位 - 反复减除数直到余数小于除数
                while (temp_segment.compareAbsWith(temp_divisor) >= 0) {
                    temp_segment = temp_segment.subtractAbsolute(temp_segment, temp_divisor);
                    count++;
                }
                quotient += to_string(count);
                // 更新当前段为新的余数
                current_segment = temp_segment.toIntegerString();
            }
        }
        
        // 处理特殊情况 - 如果商为空则设为"0"
        if (quotient.empty()) quotient = "0";
        
        BigNumber result(quotient);
        result.decimal_position = 0;
        return result;
    }
};

int main() {   
    int n_cases;
    cin >> n_cases;
    
    while (n_cases--) {
        char op_char;
        string s1_in, s2_in;
        cin >> op_char >> s1_in >> s2_in;
        
        BigNumber num1(s1_in);
        BigNumber num2(s2_in);
        BigNumber result;
        
        switch (op_char) {
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
                break;
            default:
                result = BigNumber("ERROR");
        }
        
        cout << result.toString() << endl;
    }
    // string a0  = "66,666,666,666,666,666,666,666,666,666,666.6666666666"; string b0 ="-2" ;
    // BigNumber a(a0); BigNumber b(b0);
    // cout << b.compareAbsWith(a);

    return 0;
}
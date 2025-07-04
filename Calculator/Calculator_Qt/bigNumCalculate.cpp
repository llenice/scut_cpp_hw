#include "bigNumCal.h"
#include <vector>    // Required for std::vector in multiplyLists & compareLists
#include <algorithm> // Required for std::remove, std::reverse, std::max
#include <stdexcept> // Required for std::runtime_error
#include <iostream>  // For potential debugging, can be removed if not used

// Using std namespace for convenience, or qualify (e.g., std::string)
using namespace std;

BigNumber::BigNumber() : head(nullptr), is_negative(false) {
    head = new ListNode(0);  // 初始化为0
}

BigNumber::BigNumber(const string& str) : head(nullptr), is_negative(false) {
    parseFromString(str);
}

BigNumber::BigNumber(const BigNumber& other) : head(nullptr), is_negative(other.is_negative) {
    copyFrom(other);
}

BigNumber::~BigNumber() {
    clear();
}

BigNumber& BigNumber::operator=(const BigNumber& other) {
    if (this != &other) {
        clear();
        is_negative = other.is_negative;
        copyFrom(other);
    }
    return *this;
}

void BigNumber::clear() {
    while (head) {
        ListNode* temp = head;
        head = head->next;
        delete temp;
    }
    head = nullptr; // Ensure head is null after clearing
}

void BigNumber::copyFrom(const BigNumber& other) {
    if (!other.head) {
        // Ensure current list is cleared and set to 0 if other is invalid/empty
        clear();
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

void BigNumber::parseFromString(const string& str_in) {
    clear(); // Clear existing data

    string s = str_in;
    // 移除所有逗号
    s.erase(remove(s.begin(), s.end(), ','), s.end());

    if (s.empty()) s = "0";

    is_negative = false;
    if (!s.empty() && s[0] == '-') {
        is_negative = true;
        s = s.substr(1);
    }

    if (s.empty()) s = "0"; // Handle case where string was just "-"

    // 移除前导零
    size_t first_digit = s.find_first_not_of('0');
    if (string::npos == first_digit) { // All zeros or empty after sign removal
        s = "0";
        is_negative = false; // "0" is not negative
    } else {
        s = s.substr(first_digit);
    }

    if (s == "0") is_negative = false; // Normalize -0 to 0

    // 从右到左按BASE_DIGITS位分组创建链表节点
    // The original C_list.cpp stores digits with the least significant at the head.
    // So, we process the string from right to left, creating nodes and prepending them,
    // or create them in order and then reverse the list, or build in reverse order.
    // The original code built it such that head points to the LSB part.
    // Let's stick to head being LSB.

    head = nullptr; // Will be set to the first (least significant) segment
    ListNode* current_tail = nullptr;


    int len = s.length();
    for (int i = len; i > 0; i -= BASE_DIGITS) {
        int start = std::max(0, i - BASE_DIGITS);
        string segment_str = s.substr(start, i - start);
        int value = 0;
        try {
            value = stoi(segment_str);
        } catch (const std::out_of_range& oor) {
            // This should ideally not happen if BASE_DIGITS is small enough
            // and input string contains only digits.
            // For robustness, handle or log. Here, we might default to 0 or rethrow.
            // Or, if string is too large for int, it means BASE_DIGITS is too large.
            // Given BASE_DIGITS = 4, segment_str max "9999", fits in int.
            // This catch is more for stoi on non-numeric or empty strings,
            // but previous checks should prevent that.
            throw std::runtime_error("Error converting segment to integer: " + segment_str);
        }


        ListNode* newNode = new ListNode(value);
        if (!head) { // First node (least significant part)
            head = newNode;
            current_tail = head;
        } else { // Add to the end of the list (which means higher significant part)
            current_tail->next = newNode;
            current_tail = newNode;
        }
    }

    if (!head) { // If the string was empty or resulted in no digits (e.g. "0")
        head = new ListNode(0);
    }
    // No need to reverse if we build from LSB to MSB and append to tail.
}


string BigNumber::toStringPlain() const {
    if (!head) return "0"; // Should not happen if constructor ensures head is not null

    if (isZero()) { // Handles the case where head is [0]->null
        return "0";
    }

    string result_str = "";
    if (is_negative) result_str += "-";

    vector<string> segments;
    ListNode* cur = head;
    while (cur) {
        segments.push_back(std::to_string(cur->data));
        cur = cur->next;
    }
    std::reverse(segments.begin(), segments.end()); // Higher order segments are at the end of vector now

    bool first_segment = true;
    for (const string& seg : segments) {
        if (first_segment) {
            result_str += seg;
            first_segment = false;
        } else {
            string temp_seg = seg;
            while (temp_seg.length() < BASE_DIGITS) {
                temp_seg = "0" + temp_seg;
            }
            result_str += temp_seg;
        }
    }
    return result_str;
}

string BigNumber::toString() const {
    return addThousandSeparators(toStringPlain());
}

string BigNumber::addThousandSeparators(const string& num_str) const {
    string s = num_str;
    string sign = "";

    if (!s.empty() && s[0] == '-') {
        sign = "-";
        s = s.substr(1);
    }

    if (s == "0") return "0"; // Handles "0" and "-0" (which should be normalized to "0")

    string formatted_num = "";
    int digit_count = 0;
    for (int i = s.length() - 1; i >= 0; i--) {
        formatted_num += s[i];
        digit_count++;
        if (digit_count == 3 && i != 0) {
            formatted_num += ',';
            digit_count = 0;
        }
    }
    std::reverse(formatted_num.begin(), formatted_num.end());
    return sign + formatted_num;
}

bool BigNumber::isZero() const {
    if (!head) return true; // Or handle as an error/exception
    ListNode* cur = head;
    while (cur) {
        if (cur->data != 0) return false;
        cur = cur->next;
    }
    // Special check: if head is [0]->null and is_negative is true, it's still considered zero.
    // The parseFromString and other operations should normalize -0 to 0.
    return true;
}

void BigNumber::removeLeadingZeros() {
    // This function in the original code seems to remove trailing zero *nodes*
    // because the list is stored LSB first. "Leading zeros" in the number
    // correspond to the most significant nodes if they are zero.
    // The goal is to ensure the most significant node (last in list) is not 0, unless number is 0.

    if (!head || !head->next) return; // Single node or empty list, nothing to trim

    ListNode* current = head;
    ListNode* last_non_zero_node_prev = nullptr; // Tracks the node *before* the potential new end of list

    // Traverse to find the effective end of the list (last non-zero node)
    // This is tricky because the list is LSB first. We need to find the MSB non-zero.
    // Let's convert to string, parse again for canonical form, or rethink.
    // The original logic was:
    // 1. Find the last node (MSB).
    // 2. If it's 0 and not the only node, remove it and repeat.

    // Re-implementing based on the idea of removing trailing zero nodes (MSB zeros)
    // while keeping at least one node if the number is 0.

    // First, reverse the list to operate on MSB easily, then reverse back.
    // Or, more simply, find the last node that isn't zero.
    ListNode *prev_to_last_non_zero = nullptr;
    ListNode *last_non_zero = nullptr;
    ListNode *iter = head;
    ListNode *prev_iter = nullptr;

    while(iter != nullptr){
        if(iter->data != 0){
            last_non_zero = iter;
            prev_to_last_non_zero = prev_iter;
        }
        prev_iter = iter;
        iter = iter->next;
    }

    if(last_non_zero == nullptr) { // All nodes are zero
        if(head->data == 0 && head->next != nullptr) { // Multiple zero nodes e.g. 0 -> 0 -> 0
            clear();
            head = new ListNode(0);
        }
        // If only one node [0]->null, it's fine. is_negative should be false.
        if(isZero()) is_negative = false;
        return;
    }

    // last_non_zero is the new most significant node. Delete nodes after it.
    if(last_non_zero->next != nullptr){
        ListNode* current_to_delete = last_non_zero->next;
        last_non_zero->next = nullptr; // Truncate the list
        while(current_to_delete != nullptr){
            ListNode* temp = current_to_delete;
            current_to_delete = current_to_delete->next;
            delete temp;
        }
    }
    if(isZero()) is_negative = false; // Final check
}


// Helper functions implementations

BigNumber addLists(const BigNumber& a, const BigNumber& b) {
    BigNumber result;
    result.clear(); // Start with an empty list for result

    ListNode* p1 = a.head;
    ListNode* p2 = b.head;
    ListNode* result_tail = nullptr;
    int carry = 0;

    while (p1 || p2 || carry) {
        int sum = carry;
        if (p1) {
            sum += p1->data;
            p1 = p1->next;
        }
        if (p2) {
            sum += p2->data;
            p2 = p2->next;
        }

        ListNode* newNode = new ListNode(sum % BASE);
        carry = sum / BASE;

        if (!result.head) {
            result.head = newNode;
            result_tail = result.head;
        } else {
            result_tail->next = newNode;
            result_tail = newNode;
        }
    }
    if (!result.head) { // If both a and b were zero (or empty)
        result.head = new ListNode(0);
    }
    return result;
}

BigNumber subtractLists(const BigNumber& a, const BigNumber& b) {
    // Assumes a >= b and both are positive. Sign handling is done by BigReal or higher level.
    BigNumber result;
    result.clear();

    ListNode* p1 = a.head; // Larger number
    ListNode* p2 = b.head; // Smaller number
    ListNode* result_tail = nullptr;
    int borrow = 0;

    while (p1) { // Iterate through the larger number
        int val1 = p1->data;
        int val2 = 0;
        if (p2) {
            val2 = p2->data;
            p2 = p2->next;
        }

        int diff = val1 - val2 - borrow;
        if (diff < 0) {
            diff += BASE;
            borrow = 1;
        } else {
            borrow = 0;
        }

        ListNode* newNode = new ListNode(diff);
        if (!result.head) {
            result.head = newNode;
            result_tail = result.head;
        } else {
            result_tail->next = newNode;
            result_tail = newNode;
        }
        p1 = p1->next;
    }

    if (!result.head) {
        result.head = new ListNode(0);
    }
    result.removeLeadingZeros(); // Crucial for canonical form
    return result;
}

int compareLists(const BigNumber& a, const BigNumber& b) {
    // This comparison ignores signs.
    // Convert to plain string for easier comparison of magnitude,
    // or compare node by node from MSB.
    // The original code counts nodes then compares from MSB.

    vector<int> v_a, v_b;
    ListNode* cur = a.head;
    while(cur) { v_a.push_back(cur->data); cur = cur->next; }
    cur = b.head;
    while(cur) { v_b.push_back(cur->data); cur = cur->next; }

    // Remove leading zeros from vectors for fair comparison of length
    // This means removing zeros from the *end* of the vector as it's LSB first
    while(v_a.size() > 1 && v_a.back() == 0) v_a.pop_back();
    while(v_b.size() > 1 && v_b.back() == 0) v_b.pop_back();


    if (v_a.size() > v_b.size()) return 1;
    if (v_a.size() < v_b.size()) return -1;

    // Same number of significant "digits" (nodes)
    for (int i = v_a.size() - 1; i >= 0; --i) { // Compare from MSB
        if (v_a[i] > v_b[i]) return 1;
        if (v_a[i] < v_b[i]) return -1;
    }
    return 0; // Equal
}

BigNumber multiplyLists(const BigNumber& a, const BigNumber& b) {
    BigNumber result_bn; // Will be initialized to 0
    if (a.isZero() || b.isZero()) {
        return result_bn; // Return 0
    }
    result_bn.clear(); // Clear the default 0 node

    vector<int> n1, n2;
    ListNode* cur = a.head;
    while(cur) { n1.push_back(cur->data); cur = cur->next; } // n1 is LSB first
    cur = b.head;
    while(cur) { n2.push_back(cur->data); cur = cur->next; } // n2 is LSB first

    vector<long long> res_vec(n1.size() + n2.size(), 0);

    for (size_t i = 0; i < n1.size(); ++i) {
        for (size_t j = 0; j < n2.size(); ++j) {
            res_vec[i + j] += (long long)n1[i] * n2[j];
        }
    }

    long long carry = 0;
    for (size_t i = 0; i < res_vec.size(); ++i) {
        res_vec[i] += carry;
        carry = res_vec[i] / BASE;
        res_vec[i] %= BASE;
    }
    // If carry remains, it means res_vec was not large enough,
    // but it should be due to n1.size + n2.size.
    // However, if the last element itself produces a carry:
    // while (carry > 0) { // This loop was in original, check if needed with current res_vec size
    //    res_vec.push_back(carry % BASE);
    //    carry /= BASE;
    // }
    // The initial size n1.size() + n2.size() should be enough to hold the product
    // and the final carry will be part of the last element if res_vec[res_vec.size()-1]
    // For example, 99 * 99 with BASE 100. n1=[99], n2=[99]. res_vec size 2.
    // res_vec[0] = 99*99 = 9801.
    // i=0: res_vec[0]=9801. carry = 98. res_vec[0]=1.
    // i=1: res_vec[1]=0+98=98. carry = 0. res_vec[1]=98.
    // Result: [1, 98] -> 9801. Correct.

    ListNode* result_tail = nullptr;
    for (size_t i = 0; i < res_vec.size(); ++i) {
        // Skip leading zeros in the result vector if they are at the most significant end
        // and the vector is not representing just "0".
        if (i == res_vec.size() - 1 && res_vec[i] == 0 && result_bn.head != nullptr) {
            // If the most significant part is 0 and we already have parts, don't add it
            // unless it's the only part and it's 0.
            if (result_bn.head == nullptr && res_vec[i] == 0) { // Case for 0 * X = 0
                // Let it add the zero node.
            } else if (res_vec[i] == 0 && i > 0) { // Don't add trailing zero segments if result is not just 0
                bool all_zeros_so_far = true;
                for(size_t k=0; k<i; ++k) if(res_vec[k] != 0) all_zeros_so_far = false;
                if(all_zeros_so_far && res_vec[i] == 0 && i == res_vec.size() -1) {
                    // This is to handle 0 result
                } else if (res_vec[i] == 0 && result_bn.head == nullptr && i < res_vec.size() -1) {
                    // if it's like [0,0,5] we are at i=0, res_vec[0]=0, head is null. We should add it.
                } else if (res_vec[i] == 0 && result_bn.head != nullptr && i == res_vec.size() -1) {
                    // if it's like [5,0,0] and we are at the last 0, don't add it.
                    // This is complex. removeLeadingZeros should handle it.
                }
            }
        }

        ListNode* newNode = new ListNode(static_cast<int>(res_vec[i]));
        if (!result_bn.head) {
            result_bn.head = newNode;
            result_tail = result_bn.head;
        } else {
            result_tail->next = newNode;
            result_tail = newNode;
        }
    }

    if (!result_bn.head) { // Should only happen if input vectors were empty, or product is 0
        result_bn.head = new ListNode(0);
    }
    result_bn.removeLeadingZeros();
    return result_bn;
}


BigNumber divideLists(const BigNumber& dividend_bn, const BigNumber& divisor_bn) {
    // Performs integer division: dividend / divisor.
    // Assumes both are positive.
    BigNumber quotient_bn; // Initializes to 0

    if (divisor_bn.isZero()) {
        throw std::runtime_error("Division by zero in divideLists");
    }
    if (dividend_bn.isZero()) {
        return quotient_bn; // 0 / X = 0
    }
    if (compareLists(dividend_bn, divisor_bn) < 0) {
        return quotient_bn; // Dividend < Divisor, so integer quotient is 0
    }
    if (compareLists(dividend_bn, divisor_bn) == 0) {
        quotient_bn.parseFromString("1"); // X / X = 1
        return quotient_bn;
    }

    // Simplified approach using string conversion for long division, as in original.
    // A direct list-based long division is more complex to implement here.
    string dividend_str = dividend_bn.toStringPlain();
    string divisor_str = divisor_bn.toStringPlain();

    string quotient_str = "";
    BigNumber current_segment_bn; // Represents the current part of the dividend being processed
    current_segment_bn.parseFromString("0"); // Start with 0

    // Iterate through digits of the dividend string (which is MSB first)
    for (char digit_char : dividend_str) {
        // Append current digit to current_segment_bn
        // This is tricky with BigNumber. Let's use string for current_segment for now for simplicity
        // as in the original code.
        string current_segment_str = current_segment_bn.toStringPlain();
        if (current_segment_str == "0") current_segment_str = ""; // Avoid "05"
        current_segment_str += digit_char;
        current_segment_bn.parseFromString(current_segment_str);

        if (compareLists(current_segment_bn, divisor_bn) < 0) {
            if (!quotient_str.empty()) { // Avoid leading "0" if first segment is smaller
                quotient_str += "0";
            }
        } else {
            int count = 0;
            while (compareLists(current_segment_bn, divisor_bn) >= 0) {
                current_segment_bn = subtractLists(current_segment_bn, divisor_bn);
                count++;
            }
            quotient_str += std::to_string(count);
        }
    }

    if (quotient_str.empty()) { // e.g. 1 / 2
        quotient_str = "0";
    }
    quotient_bn.parseFromString(quotient_str);
    return quotient_bn;
}




BigReal::BigReal(const string& s) : fractional_digits(0), is_negative(false) {
    // integer_part and fractional_part are default constructed by BigNumber() to "0"
    if (s == "ERROR") {
        error_message = "ERROR";
        return;
    }
    parse(s);
}

void BigReal::parse(const string& s_in) {
    // Reset state
    integer_part.parseFromString("0");
    fractional_part.parseFromString("0");
    fractional_digits = 0;
    is_negative = false;
    error_message = "";

    string str = s_in;
    // 移除所有逗号
    str.erase(remove(str.begin(), str.end(), ','), str.end());

    if (str.empty()) str = "0";

    if (!str.empty() && str[0] == '-') {
        is_negative = true;
        str = str.substr(1);
    }

    if (str.empty() || str == "." || str == "-") str = "0"; // Handle cases like "." or just "-"

    size_t dot_pos = str.find('.');
    if (dot_pos == string::npos) {
        // 整数
        integer_part.parseFromString(str);
        // fractional_part remains "0", fractional_digits remains 0
    } else {
        // 有小数部分
        string int_str_part = str.substr(0, dot_pos);
        string frac_str_part = str.substr(dot_pos + 1);

        if (int_str_part.empty()) int_str_part = "0";
        // Remove trailing zeros from fractional part string before parsing to BigNumber,
        // but keep track of original length for fractional_digits.
        // However, C_list stores it as is and fractional_digits is its length.

        integer_part.parseFromString(int_str_part);

        if (frac_str_part.empty()) {
            fractional_part.parseFromString("0");
            fractional_digits = 0; // Or based on original string if "1." means "1.0"
                // C_list seems to treat "1." as integer 1.
                // If "1.0", frac_str_part is "0", frac_digits is 1.
        } else {
            fractional_part.parseFromString(frac_str_part);
            fractional_digits = frac_str_part.length();
        }
    }

    // Signs of BigNumber components are always false; BigReal holds the overall sign.
    integer_part.is_negative = false;
    fractional_part.is_negative = false;

    // Normalize zero
    if (integer_part.isZero() && fractional_part.isZero()) {
        is_negative = false;
        // fractional_digits might remain if input was "0.00"
        // For consistency, if value is 0, perhaps fractional_digits should be 0 too.
        // C_list's toString for "0.00" seems to output "0".
        // Let's keep fractional_digits as parsed unless it's truly zero value.
        // If parse("0.00"), int_part="0", frac_part="00", frac_digits=2.
        // Then frac_part.parseFromString("00") becomes BigNumber("0").
        // So integer_part.isZero() and fractional_part.isZero() is true.
    }
    // If integer_part became "0" (e.g. from ".5"), ensure it's not negative.
    if (integer_part.isZero() && integer_part.is_negative) {
        integer_part.is_negative = false;
    }
}


string BigReal::toString() const {
    if (!error_message.empty()) return error_message;

    string res_str = "";
    if (is_negative && !(integer_part.isZero() && fractional_part.isZero())) {
        res_str += "-";
    }

    res_str += integer_part.toString(); // This already handles thousand separators

    if (fractional_digits > 0) { // Only add decimal point if there were fractional digits parsed
        string frac_plain_str = fractional_part.toStringPlain(); // Get plain digits of fractional part

        // Pad with leading zeros if parsed fractional_part was like "012" for ".012"
        // and fractional_part.toStringPlain() gave "12". No, fractional_part stores "012" as number 12.
        // fractional_digits is the key.

        string frac_to_display = frac_plain_str;
        if (frac_plain_str == "0" && fractional_digits > 0) { // e.g. input was 1.00, frac_part is 0, frac_digits is 2
            frac_to_display = string(fractional_digits, '0');
        } else {
            // Ensure frac_to_display has fractional_digits length by padding leading zeros
            // if the number stored in fractional_part has fewer digits than fractional_digits.
            // Example: input "0.00123". fractional_part stores "123". fractional_digits = 5.
            // frac_plain_str = "123". Need "00123".
            string temp_frac_str = fractional_part.toStringPlain(); // Get the number "123"
            if (temp_frac_str == "0" && fractional_part.isZero()) { // if fractional part is truly 0
                // if fractional_digits > 0, means we want to show .0, .00 etc.
                if(fractional_digits > 0) {
                    frac_to_display = string(fractional_digits, '0');
                } else { // Should not happen if fractional_digits > 0 condition is met
                    frac_to_display = "";
                }
            } else { // fractional part is non-zero
                frac_to_display = temp_frac_str;
                while(frac_to_display.length() < fractional_digits){
                    frac_to_display = "0" + frac_to_display;
                }
            }
        }


        // Remove trailing zeros from the display string of fractional part,
        // UNLESS all are zeros and we want to show them (e.g. "1.00" should show ".00" if desired)
        // The C_list.cpp logic for toString was:
        // while (!frac_str.empty() && frac_str.back() == '0') frac_str.pop_back();
        // This means 1.230 will be 1.23, and 1.00 will be 1.
        // Let's replicate that.

        string final_frac_display = frac_to_display;
        // If fractional_part itself is zero, frac_to_display might be "0", "00" etc.
        // If fractional_part is non-zero, e.g. "12300" (for .12300, frac_digits=5)
        // then frac_to_display is "12300". We want "123".

        if (!fractional_part.isZero()) { // Only trim if fractional part isn't just zeros
            while(!final_frac_display.empty() && final_frac_display.back() == '0'){
                final_frac_display.pop_back();
            }
        } else { // Fractional part is zero.
            // If original input was "1.0", frac_digits=1, frac_part=0. frac_to_display="0".
            // Original code would make final_frac_display empty. So "1." -> "1".
            // If original "1.00", frac_digits=2, frac_part=0. frac_to_display="00".
            // Original code would make final_frac_display empty. So "1.00" -> "1".
            // This seems to be the behavior.
            while(!final_frac_display.empty() && final_frac_display.back() == '0'){
                final_frac_display.pop_back();
            }
        }


        if (!final_frac_display.empty()) {
            res_str += ".";
            res_str += final_frac_display;
        } else if (fractional_digits > 0 && fractional_part.isZero() && res_str.find('.') == string::npos) {
            // This case is if input was "0.0", it should display "0", not "0."
            // If input was "1.0", it should display "1".
            // The original code's logic seems to achieve this by frac_str becoming empty.
        }
    }

    if (res_str == "-" || res_str.empty()) res_str = "0"; // Should be covered by BigNumber::toString if int_part is 0
    if (res_str == "-0") res_str = "0"; // Normalize -0
    if (integer_part.isZero() && fractional_part.isZero() && fractional_digits == 0) return "0";


    return res_str;
}


bool BigReal::isAbsLessThan1e_6() const {
    if (!error_message.empty()) return false;

    BigReal abs_val = *this;
    abs_val.is_negative = false;

    // Create BigReal for 1e-6 (0.000001)
    BigReal threshold_val("0.000001");

    // Compare abs_val with threshold_val
    // This requires a BigReal comparison function.
    // For simplicity, let's use the original logic if it doesn't rely on full BigReal comparison.

    // Original logic:
    if (!integer_part.isZero()) return false; // If |int| >= 1, then not < 1e-6

    // Now integer_part is 0. Check fractional_part.
    // If fractional_digits < 6, it could be like 0.1 (false), 0.00001 (true)
    // If fractional_digits >= 6:
    //   0.000000... (true)
    //   0.000001... (false if > 1 at 6th or later significant digit)
    //   0.000000123 (true)

    // Convert fractional part to a string padded to at least 6 digits
    string frac_str_plain = fractional_part.toStringPlain();

    // We need to consider the actual value. "0.0000001" has fractional_digits = 7, frac_str_plain = "1"
    // "0.00001" has fractional_digits = 5, frac_str_plain = "1"

    if (fractional_digits >= 7) { // e.g., 0.0000001 or smaller
        // Check if all first 6 decimal places are zero.
        // This means fractional_part's string representation, when padded to fractional_digits,
        // must have zeros for a certain prefix.
        // Example: 0.0000001. fractional_digits=7. fractional_part="1".
        // Effective number string: "0000001"
        // First 6 are "000000". This is true.

        // If fractional_part is "0", then it's 0.000... which is < 1e-6.
        if (fractional_part.isZero()) return true;

        // If fractional_digits is, say, 7, and fractional_part is "1" (value 0.0000001)
        // We need to check if the number is < 0.000001
        // This means we need to look at the first 6 effective decimal places.
        // If fractional_digits is large, say 10, and value is 0.0000000001
        // fractional_part is "1", effective string "0000000001". First 6 are zero. True.

        // A simpler way: construct the number as a plain string "0.xxxxxxxx"
        string full_frac_string = fractional_part.toStringPlain();
        while(full_frac_string.length() < fractional_digits) {
            full_frac_string = "0" + full_frac_string;
        }
        // Now full_frac_string is like "001" for 0.001 (frac_digits=3)
        // or "123" for 0.123 (frac_digits=3)

        for(int i=0; i < std::min(6, (int)full_frac_string.length()); ++i) {
            if (i < fractional_digits) { // only check within actual decimal places
                if (full_frac_string[i] != '0') return false; // Non-zero digit found in first 6 places
            } else { // Padded beyond actual digits, effectively zero
                break;
            }
        }
        // If we pass the loop, it means the first min(6, fractional_digits) are zero.
        // If fractional_digits < 6, say 3, and they are "000", then true.
        // If fractional_digits >= 6 and first 6 are "0", then true.
        if (fractional_digits < 6) { // e.g. 0.000, all checked were 0
            bool all_zero_in_frac = true;
            for(char c : full_frac_string) if(c != '0') all_zero_in_frac = false;
            return all_zero_in_frac;
        }
        return true; // First 6 (or fewer if frac_digits < 6) are zero.

    } else { // fractional_digits < 6 (e.g., 0 to 5)
        // Example: 0.1 (fd=1), 0.01 (fd=2), 0.00001 (fd=5)
        // All these are > 0.000001 unless the number is exactly 0.
        return fractional_part.isZero();
    }
}


// Helper functions for BigReal

BigReal multiply(const BigReal& a, const BigReal& b) {
    if (!a.error_message.empty()) return BigReal("ERROR");
    if (!b.error_message.empty()) return BigReal("ERROR");

    // Determine sign of result
    bool result_is_negative = a.is_negative ^ b.is_negative;

    // Convert a and b to BigNumbers by concatenating integer and fractional parts
    string a_full_str = a.integer_part.toStringPlain();
    if (a.fractional_digits > 0) {
        string temp_frac = a.fractional_part.toStringPlain();
        while(temp_frac.length() < a.fractional_digits) temp_frac = "0" + temp_frac;
        a_full_str += temp_frac;
    }

    string b_full_str = b.integer_part.toStringPlain();
    if (b.fractional_digits > 0) {
        string temp_frac = b.fractional_part.toStringPlain();
        while(temp_frac.length() < b.fractional_digits) temp_frac = "0" + temp_frac;
        b_full_str += temp_frac;
    }

    BigNumber a_bn(a_full_str);
    BigNumber b_bn(b_full_str);

    BigNumber product_bn = multiplyLists(a_bn, b_bn);

    int total_fractional_digits = a.fractional_digits + b.fractional_digits;
    string product_str_plain = product_bn.toStringPlain();

    BigReal result_br; // Default constructor makes it "0"
    if (product_str_plain == "0") { // Handle if product is exactly 0
        result_br.parse("0"); // Already done by constructor
    } else if (total_fractional_digits == 0) {
        result_br.parse(product_str_plain);
    } else {
        if (product_str_plain.length() <= total_fractional_digits) {
            // Result is purely fractional or 0.xxx
            string zeros(total_fractional_digits - product_str_plain.length(), '0');
            result_br.parse("0." + zeros + product_str_plain);
        } else {
            // Result has an integer part
            size_t split_point = product_str_plain.length() - total_fractional_digits;
            string int_part = product_str_plain.substr(0, split_point);
            string frac_part = product_str_plain.substr(split_point);
            result_br.parse(int_part + "." + frac_part);
        }
    }

    result_br.is_negative = result_is_negative;
    // Normalize if result is zero
    if (result_br.integer_part.isZero() && result_br.fractional_part.isZero()) {
        result_br.is_negative = false;
    }
    return result_br;
}

BigReal divide(const BigReal& a, const BigReal& b) {
    if (!a.error_message.empty()) return BigReal("ERROR");
    if (!b.error_message.empty()) return BigReal("ERROR");

    if (b.isAbsLessThan1e_6()) { // Check for division by number very close to zero
        return BigReal("ERROR"); // Division by zero or too small number
    }

    if (a.integer_part.isZero() && a.fractional_part.isZero()) {
        return BigReal("0"); // 0 / X = 0 (where X is not zero)
    }

    bool result_is_negative = a.is_negative ^ b.is_negative;

    // Convert a and b to BigNumbers, scale 'a' for precision
    string a_full_str = a.integer_part.toStringPlain();
    if (a.fractional_digits > 0) {
        string temp_frac = a.fractional_part.toStringPlain();
        while(temp_frac.length() < a.fractional_digits) temp_frac = "0" + temp_frac;
        a_full_str += temp_frac;
    }

    string b_full_str = b.integer_part.toStringPlain();
    if (b.fractional_digits > 0) {
        string temp_frac = b.fractional_part.toStringPlain();
        while(temp_frac.length() < b.fractional_digits) temp_frac = "0" + temp_frac;
        b_full_str += temp_frac;
    }

    // Precision: Add zeros to 'a' to simulate decimal division.
    // The number of zeros determines precision of fractional part of quotient.
    // Original code used precision_needed = 11. This means 10 decimal places after rounding.
    int precision_zeros = 10 + b.fractional_digits; // Need enough to cover b's decimals and get 10 more for quotient
    // More robust: precision_zeros = 10 (for result) + a.fractional_digits (original) + b.fractional_digits (to make b integer like)
    // Let's use a fixed number of precision zeros for now, similar to C_list.cpp's effective precision.
    // C_list.cpp added 11 zeros, then adjusted exponent by -10. So, 10 decimal places of precision.

    int scale_factor_zeros = 10 + ( (b.fractional_digits > a.fractional_digits) ? b.fractional_digits : 0 ); // Heuristic
    // A simpler approach from C_list.cpp: add 11 zeros to a_full_str.
    // The effective number of decimal places in the quotient will be
    // (a.frac_digits + added_zeros) - b.frac_digits. We want this to be around 10.
    // So, added_zeros = 10 - a.frac_digits + b.frac_digits.
    // C_list.cpp used fixed 11 zeros and then adjusted final_exp.

    for (int i = 0; i < 11; ++i) { // Add 11 zeros for precision (10 places + 1 for rounding)
        a_full_str += "0";
    }

    BigNumber dividend_bn(a_full_str);
    BigNumber divisor_bn(b_full_str);

    if (divisor_bn.isZero()){ // Should have been caught by isAbsLessThan1e_6, but as a safeguard
        return BigReal("ERROR");
    }

    BigNumber quotient_bn = divideLists(dividend_bn, divisor_bn); // Integer division
    string quotient_str_plain = quotient_bn.toStringPlain();

    // Rounding: last digit of quotient_str_plain is for rounding (11th effective decimal)
    char rounding_digit = '0';
    if (!quotient_str_plain.empty() && quotient_str_plain != "0") {
        rounding_digit = quotient_str_plain.back();
        quotient_str_plain.pop_back();
        if (quotient_str_plain.empty()) quotient_str_plain = "0";
    }

    if (rounding_digit >= '5') {
        BigNumber temp_quotient(quotient_str_plain);
        BigNumber one("1");
        temp_quotient = addLists(temp_quotient, one);
        quotient_str_plain = temp_quotient.toStringPlain();
    }

    // Determine position of decimal point in quotient_str_plain
    // Original exponent of a: -a.fractional_digits
    // Original exponent of b: -b.fractional_digits
    // Exponent of a_full_str (scaled a): -a.fractional_digits - 11 (due to added zeros)
    // Exponent of b_full_str (scaled b): -b.fractional_digits
    // Exponent of quotient_bn = Exp(a_full_str) - Exp(b_full_str)
    // = (-a.fractional_digits - 11) - (-b.fractional_digits)
    // = b.fractional_digits - a.fractional_digits - 11
    // This is the exponent for quotient_str_plain (which has 10 effective decimal places after rounding).
    // So, number of decimal places in result is -(exponent) = a.fd - b.fd + 11 -1(for rounding) = a.fd - b.fd + 10
    int num_decimal_places_in_result = a.fractional_digits - b.fractional_digits + 10;


    BigReal result_br;
    if (quotient_str_plain == "0") {
        result_br.parse("0");
    } else if (num_decimal_places_in_result <= 0) { // Result is a large integer or integer
        string zeros_to_add(-num_decimal_places_in_result, '0');
        result_br.parse(quotient_str_plain + zeros_to_add);
    } else { // Result has decimal places
        if (quotient_str_plain.length() <= num_decimal_places_in_result) {
            // Result is purely fractional or 0.xxx
            string zeros_prefix(num_decimal_places_in_result - quotient_str_plain.length(), '0');
            result_br.parse("0." + zeros_prefix + quotient_str_plain);
        } else {
            // Result has an integer part
            size_t split_point = quotient_str_plain.length() - num_decimal_places_in_result;
            string int_part = quotient_str_plain.substr(0, split_point);
            string frac_part = quotient_str_plain.substr(split_point);
            result_br.parse(int_part + "." + frac_part);
        }
    }

    result_br.is_negative = result_is_negative;
    if (result_br.integer_part.isZero() && result_br.fractional_part.isZero()) {
        result_br.is_negative = false; // Normalize zero
    }

    return result_br;
}

BigReal add(const BigReal& a, const BigReal& b) {
    if (!a.error_message.empty()) return BigReal("ERROR");
    if (!b.error_message.empty()) return BigReal("ERROR");

    // 如果符号相同，进行加法
    if (a.is_negative == b.is_negative) {
        int max_frac_digits = max(a.fractional_digits, b.fractional_digits);

        string a_str = a.integer_part.toStringPlain();
        string b_str = b.integer_part.toStringPlain();

        // 只有当存在小数部分时才处理小数对齐
        if (max_frac_digits > 0) {
            string a_frac = a.fractional_part.toStringPlain();
            string b_frac = b.fractional_part.toStringPlain();

            // 补齐小数位数
            while (a_frac.length() < a.fractional_digits) a_frac = "0" + a_frac;
            while (b_frac.length() < b.fractional_digits) b_frac = "0" + b_frac;
            while (a_frac.length() < max_frac_digits) a_frac += "0";
            while (b_frac.length() < max_frac_digits) b_frac += "0";

            a_str += a_frac;
            b_str += b_frac;
        }

        BigNumber a_num(a_str);
        BigNumber b_num(b_str);
        BigNumber sum = addLists(a_num, b_num);

        string sum_str = sum.toStringPlain();

        BigReal result;
        if (max_frac_digits == 0) {
            result.parse(sum_str);
        } else {
            if (sum_str.length() <= max_frac_digits) {
                string zeros(max_frac_digits - sum_str.length(), '0');
                result.parse("0." + zeros + sum_str);
            } else {
                int int_len = sum_str.length() - max_frac_digits;
                string int_part = sum_str.substr(0, int_len);
                string frac_part = sum_str.substr(int_len);
                result.parse(int_part + "." + frac_part);
            }
        }

        result.is_negative = a.is_negative;
        return result;
    } else {
        // 符号不同，转换为减法
        BigReal b_copy = b;
        b_copy.is_negative = !b_copy.is_negative;
        return subtract(a, b_copy);
    }
}


// 添加缺失的减法函数
BigReal subtract(const BigReal& a, const BigReal& b) {
    if (!a.error_message.empty()) return BigReal("ERROR");
    if (!b.error_message.empty()) return BigReal("ERROR");

    // 如果符号不同，转换为加法
    if (a.is_negative != b.is_negative) {
        BigReal b_copy = b;
        b_copy.is_negative = !b_copy.is_negative;
        return add(a, b_copy);
    }

    // 符号相同，需要比较绝对值大小
    int max_frac_digits = max(a.fractional_digits, b.fractional_digits);

    string a_str = a.integer_part.toStringPlain();
    string b_str = b.integer_part.toStringPlain();

    // 只有当存在小数部分时才处理小数对齐
    if (max_frac_digits > 0) {
        string a_frac = a.fractional_part.toStringPlain();
        string b_frac = b.fractional_part.toStringPlain();

        // 补齐小数位数到实际的小数位数
        while (a_frac.length() < a.fractional_digits) a_frac = "0" + a_frac;
        while (b_frac.length() < b.fractional_digits) b_frac = "0" + b_frac;

        // 补齐到相同的最大小数位数
        while (a_frac.length() < max_frac_digits) a_frac += "0";
        while (b_frac.length() < max_frac_digits) b_frac += "0";

        a_str += a_frac;
        b_str += b_frac;
    }
    // 如果都是整数，则不添加小数部分

    BigNumber a_num(a_str);
    BigNumber b_num(b_str);

    int cmp = compareLists(a_num, b_num);
    if (cmp == 0) {
        return BigReal("0");
    }

    BigNumber diff;
    bool result_negative;

    if (cmp > 0) {
        // |a| > |b|
        diff = subtractLists(a_num, b_num);
        result_negative = a.is_negative;
    } else {
        // |a| < |b|
        diff = subtractLists(b_num, a_num);
        result_negative = !a.is_negative;
    }

    string diff_str = diff.toStringPlain();

    BigReal result;
    if (max_frac_digits == 0) {
        // 纯整数结果
        result.parse(diff_str);
    } else {
        // 有小数部分的结果
        if (diff_str.length() <= max_frac_digits) {
            string zeros(max_frac_digits - diff_str.length(), '0');
            result.parse("0." + zeros + diff_str);
        } else {
            int int_len = diff_str.length() - max_frac_digits;
            string int_part = diff_str.substr(0, int_len);
            string frac_part = diff_str.substr(int_len);
            result.parse(int_part + "." + frac_part);
        }
    }

    result.is_negative = result_negative;
    if (result.integer_part.isZero() && result.fractional_part.isZero()) {
        result.is_negative = false;
    }

    return result;
}

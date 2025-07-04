#ifndef BIGNUMCAL_H
#define BIGNUMCAL_H
#include <string>   // For std::string
#include <vector>   // For std::vector
#include <algorithm> // For std::remove, std::reverse, std::max, std::min
#include <stdexcept> // For std::runtime_error
#include <iostream>  // For std::cout, std::endl (primarily for debugging, can be removed if not needed directly here)
#include <sstream>   // For std::ostringstream (if used for to_string alternatives or formatting)


// 每个节点存储的位数（基数为10^BASE_DIGITS）
const int BASE_DIGITS = 4;  // 每个节点存储4位数字
const int BASE = 10000;     // 基数为10000

/**
 * 链表节点定义
 * 每个节点存储一个整数，表示BASE_DIGITS位数字
 */
struct ListNode {
    int data;           // 存储的数字（0 <= data < BASE）
    ListNode* next;     // 指向下一个节点的指针

    ListNode(int val = 0) : data(val), next(nullptr) {}
};

class BigNumber {
public:
    ListNode* head;     // 链表头指针（指向最低位）
    bool is_negative;   // 是否为负数

    BigNumber();
    BigNumber(const std::string& str);
    BigNumber(const BigNumber& other);
    ~BigNumber();
    BigNumber& operator=(const BigNumber& other);

    void clear();
    void copyFrom(const BigNumber& other);
    void parseFromString(const std::string& str);
    std::string toStringPlain() const;
    std::string toString() const;
    std::string addThousandSeparators(const std::string& num) const;
    bool isZero() const;
    void removeLeadingZeros();
};

// Helper functions for BigNumber
BigNumber addLists(const BigNumber& a, const BigNumber& b);
BigNumber subtractLists(const BigNumber& a, const BigNumber& b); // Assumes a >= b
int compareLists(const BigNumber& a, const BigNumber& b); // 1 if a > b, -1 if a < b, 0 if a == b
BigNumber multiplyLists(const BigNumber& a, const BigNumber& b);
BigNumber divideLists(const BigNumber& dividend, const BigNumber& divisor); // Integer division


class BigReal {
public:
    BigNumber integer_part;      // 整数部分
    BigNumber fractional_part;   // 小数部分 (stored as a positive integer)
    int fractional_digits;       // 小数部分的实际位数 (e.g., for 0.00123, fractional_part is 123, fractional_digits is 5)
    bool is_negative;            // 是否为负数
    std::string error_message;   // 错误信息

    BigReal(const std::string& s = "0");
    // No need for explicit copy constructor, destructor, assignment operator if BigNumber handles them well
    // and other members are simple types. Default ones should work.

    void parse(const std::string& s);
    std::string toString() const;
    bool isAbsLessThan1e_6() const; // For checking division by ~zero
};

// Helper functions for BigReal
BigReal multiply(const BigReal& a, const BigReal& b);
BigReal divide(const BigReal& a, const BigReal& b);
BigReal subtract(const BigReal& a, const BigReal& b);
BigReal add(const BigReal& a, const BigReal& b);


#endif // BIGNUMCAL_H

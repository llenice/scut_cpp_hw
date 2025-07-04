#include "widget.h"
#include "./ui_widget.h"
#include"bigNumCal.h"
#include<bits/stdc++.h>
using namespace std;

// 用于表达式解析和求值的辅助函数（可以放在匿名命名空间或作为类的静态成员）
namespace CalculatorHelpers { // 使用命名空间避免潜在的名称冲突

// 获取运算符优先级
int getPrecedence(const string& op) {
    if (op == "+" || op == "-") return 1;
    if (op == "*" || op == "/") return 2;
    return 0; // 无效运算符或非运算符
}

// 检查字符串是否为已知的运算符
bool isOperator(const string& token) {
    return token == "+" || token == "-" || token == "*" || token == "/";
}

// 应用运算 - 调用您在 bigNumCal.h 中声明的函数
BigReal applyOp(const BigReal& b, const BigReal& a, const string& op_str) {
    // 注意参数顺序：栈顶是 a (第二个操作数)，次栈顶是 b (第一个操作数)
    // 运算通常是 b op a
    if (op_str == "+") {
        return add(b, a); // 调用 bigNumCal.h 中的 add
    } else if (op_str == "-") {
        return subtract(b, a); // 调用 bigNumCal.h 中的 subtract
    } else if (op_str == "*") {
        return multiply(b, a); // 调用 bigNumCal.h 中的 multiply
    } else if (op_str == "/") {
        return divide(b, a); // 调用 bigNumCal.h 中的 divide
    }
    // 如果 divide 会在 b 为0时抛出异常或返回带错误信息的 BigReal，这里不需要额外检查
    // 否则，您可能需要在这里添加对除以零的检查
    throw runtime_error("Invalid operator in applyOp: " + op_str);
}


// 词法分析器：将中缀表达式字符串分割为标记列表
vector<string> tokenize(const string& expression) {
    vector<string> tokens;
    string current_num_str;

    for (size_t i = 0; i < expression.length(); ++i) {
        char c = expression[i];

        if (isspace(c)) { // 跳过空格
            continue;
        }

        if (isdigit(c) || c == '.') { // 数字或小数点
            current_num_str += c;
        } else if (c == '(' || c == ')' || c == '*' || c == '/') { // 括号或高优先级运算符
            if (!current_num_str.empty()) {
                tokens.push_back(current_num_str);
                current_num_str.clear();
            }
            tokens.push_back(string(1, c));
        } else if (c == '+' || c == '-') { // 加号或减号（可能是二元或一元）
            if (!current_num_str.empty()) {
                tokens.push_back(current_num_str);
                current_num_str.clear();
            }
            // 判断是一元还是二元
            // 一元的情况：1. 表达式开头；2. 前一个标记是运算符；3. 前一个标记是左括号
            if (tokens.empty() || isOperator(tokens.back()) || tokens.back() == "(") {
                current_num_str += c; // 作为数字的一部分（带符号）
            } else {
                tokens.push_back(string(1, c)); // 作为二元运算符
            }
        } else {
            throw runtime_error(string("Invalid character in expression: ") + c);
        }
    }

    if (!current_num_str.empty()) { // 添加最后一个数字（如果存在）
        tokens.push_back(current_num_str);
    }
    return tokens;
}


// 调度场算法：中缀转后缀 (RPN)
vector<string> infixToRPN(const vector<string>& infix_tokens) {
    vector<string> output_queue;
    stack<string> operator_stack;

    for (const string& token : infix_tokens) {
        bool is_number_token = (!isOperator(token) && token != "(" && token != ")");

        if (is_number_token) {
            // 在这里可以尝试用 BigReal(token) 来验证数字格式，如果解析失败则抛异常
            // 为了简化，我们假设 tokenizer 已经生成了有效的数字字符串
            output_queue.push_back(token);
        } else if (isOperator(token)) {
            while (!operator_stack.empty() && operator_stack.top() != "(" &&
                   (getPrecedence(operator_stack.top()) > getPrecedence(token) ||
                    (getPrecedence(operator_stack.top()) == getPrecedence(token) /* && isLeftAssociative(token) */))) {
                output_queue.push_back(operator_stack.top());
                operator_stack.pop();
            }
            operator_stack.push(token);
        } else if (token == "(") {
            operator_stack.push(token);
        } else if (token == ")") {
            while (!operator_stack.empty() && operator_stack.top() != "(") {
                output_queue.push_back(operator_stack.top());
                operator_stack.pop();
            }
            if (operator_stack.empty() || operator_stack.top() != "(") {
                throw runtime_error("Mismatched parentheses: Unmatched ')' or missing '('.");
            }
            operator_stack.pop(); // 弹出 "("
        } else {
            throw runtime_error("Unknown token in Shunting-yard: " + token);
        }
    }

    while (!operator_stack.empty()) {
        if (operator_stack.top() == "(") {
            throw runtime_error("Mismatched parentheses: Unpopped '('.");
        }
        output_queue.push_back(operator_stack.top());
        operator_stack.pop();
    }
    return output_queue;
}

// 计算后缀 (RPN) 表达式
BigReal evaluateRPN(const vector<string>& rpn_tokens) {
    stack<BigReal> operand_stack;

    for (const string& token : rpn_tokens) {
        bool is_number_token = (!isOperator(token) && token != "(" && token != ")");

        if (is_number_token) {
            BigReal val(token); // BigReal 构造函数会解析字符串
            if (!val.error_message.empty()){ // 检查 BigReal 内部的解析错误
                throw runtime_error("Invalid number in RPN: '" + token + "' -> " + val.error_message);
            }
            operand_stack.push(val);
        } else if (isOperator(token)) {
            if (operand_stack.size() < 2) {
                throw runtime_error("Invalid RPN: Not enough operands for operator '" + token + "'.");
            }
            BigReal val_a = operand_stack.top(); operand_stack.pop(); // 第二个操作数
            BigReal val_b = operand_stack.top(); operand_stack.pop(); // 第一个操作数

            BigReal result_op = applyOp(val_b, val_a, token);
            if (!result_op.error_message.empty()) { // 检查运算函数返回的错误
                throw runtime_error("Operation error with '" + token + "': " + result_op.error_message);
            }
            operand_stack.push(result_op);
        } else {
            // 括号不应出现在 RPN 队列中
            throw runtime_error("Invalid token in RPN queue: " + token);
        }
    }

    if (operand_stack.size() != 1) {
        // 此处可能表示表达式格式错误，例如 "1 2" 或 "1 + "
        string error_msg = "Invalid RPN: Stack has ";
        error_msg += to_string(operand_stack.size());
        error_msg += " values at the end (expected 1).";
        if (operand_stack.empty() && !rpn_tokens.empty() && isOperator(rpn_tokens.back())) {
            error_msg += " Likely missing an operand for operator: " + rpn_tokens.back();
        } else if (operand_stack.size() > 1 && rpn_tokens.empty()) {
            error_msg += " Likely too many numbers without enough operators.";
        } else if (operand_stack.size() > 1 && !rpn_tokens.empty()) {
            // Could inspect stack and remaining tokens for more specific error
        }


        throw runtime_error(error_msg);
    }
    return operand_stack.top();
}

} // namespace CalculatorHelpers

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    this->setWindowTitle("Calculator");

    QFont f("仿宋",14);
    ui-> mainLineEdit -> setFont(f);

    QIcon con("D:\\QT\\project_ye\\Calculator\\backspace.svg");
    ui->DELETE->setIcon(con);

    ui->EQUAL->setStyleSheet("background:orange");
}

Widget::~Widget()
{
    delete ui;
}

void Widget::on_ONE_clicked()
{
    expression += "1";
    ui->mainLineEdit->setText(expression);
}

void Widget::on_TWO_clicked()
{
    expression += "2";
    ui->mainLineEdit->setText(expression);
}

void Widget::on_THREE_clicked()
{
    expression += "3";
    ui->mainLineEdit->setText(expression);
}

void Widget::on_FOUR_clicked()
{
    expression += "4";
    ui->mainLineEdit->setText(expression);
}

void Widget::on_FIVE_clicked()
{
    expression += "5";
    ui->mainLineEdit->setText(expression);
}

void Widget::on_SIX_clicked()
{
    expression += "6";
    ui->mainLineEdit->setText(expression);
}

void Widget::on_SEVEN_clicked()
{
    expression += "7";
    ui->mainLineEdit->setText(expression);
}

void Widget::on_EIGHT_clicked()
{
    expression += "8";
    ui->mainLineEdit->setText(expression);
}

void Widget::on_NINE_clicked()
{
    expression += "9";
    ui->mainLineEdit->setText(expression);
}

void Widget::on_LEFT_clicked()
{
    expression += "(";
    ui->mainLineEdit->setText(expression);
}

void Widget::on_RIGHT_clicked()
{
    expression += ")";
    ui->mainLineEdit->setText(expression);
}

void Widget::on_DIVIDE_clicked()
{
    expression += "/";
    ui->mainLineEdit->setText(expression);
}

void Widget::on_TIMES_clicked()
{
    expression += "*";
    ui->mainLineEdit->setText(expression);
}

void Widget::on_PLUS_clicked()
{
    expression += "+";
    ui->mainLineEdit->setText(expression);
}

void Widget::on_SUB_clicked()
{
    expression += "-";
    ui->mainLineEdit->setText(expression);
}

void Widget::on_ZEOR_clicked()
{
    expression += "0";
    ui->mainLineEdit->setText(expression);
}

void Widget::on_CLEAR_clicked()
{
    expression.clear();
    ui->mainLineEdit->clear();
}

void Widget::on_DELETE_clicked()
{
    expression.chop(1);
    ui->mainLineEdit->setText(expression);
}


void Widget::on_EQUAL_clicked()
{
    string expr_std_str = expression.toStdString();
    if (expr_std_str.empty()) {
        // ui->mainLineEdit->setText("0"); // 或者什么都不做
        return;
    }

    // 调试输出
    // cout << "Original expression: " << expr_std_str << endl;

    try {
        // 1. 词法分析：将中缀表达式字符串转换为标记列表
        vector<string> infix_tokens = CalculatorHelpers::tokenize(expr_std_str);

        // 调试输出
        // cout << "Infix Tokens: ";
        // for(const auto& t : infix_tokens) cout << t << " ";
        // cout << endl;

        // 2. 语法分析与转换：使用调度场算法将中缀标记转换为后缀 (RPN) 标记
        vector<string> rpn_tokens = CalculatorHelpers::infixToRPN(infix_tokens);

        // 调试输出
        // cout << "RPN Tokens: ";
        // for(const auto& t : rpn_tokens) cout << t << " ";
        // cout << endl;

        // 3. 计算后缀表达式
        BigReal final_result = CalculatorHelpers::evaluateRPN(rpn_tokens);

        // 检查 BigReal 内部是否有错误信息 (例如，除以零时 BigReal::divide 可能设置 error_message)
        if (!final_result.error_message.empty()) {
            ui->mainLineEdit->setText(QString::fromStdString(final_result.error_message));
            // expression.clear(); // 可选：出错时清空表达式
        } else {
            QString result_qstr = QString::fromStdString(final_result.toString());
            ui->mainLineEdit->setText(result_qstr);
            expression = result_qstr; // 将结果存回 expression，方便连续计算
        }

    } catch (const runtime_error& e) {
        ui->mainLineEdit->setText(QString("Error: %1").arg(e.what()));
        // expression.clear(); // 可选
    } catch (const exception& e) { // 捕获其他标准异常
        ui->mainLineEdit->setText(QString("Unexpected Error: %1").arg(e.what()));
        // expression.clear();
    } catch (...) { // 捕获所有其他类型的异常
        ui->mainLineEdit->setText("An unknown error occurred during calculation.");
        // expression.clear();
    }
}


void Widget::on_DIVIDE_2_clicked()
{
    expression += ".";
    ui->mainLineEdit->setText(expression);
}


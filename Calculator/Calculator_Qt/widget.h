#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>

QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();

private slots:
    void on_ONE_clicked();

    void on_TWO_clicked();

    void on_THREE_clicked();

    void on_FOUR_clicked();

    void on_FIVE_clicked();

    void on_SIX_clicked();

    void on_SEVEN_clicked();

    void on_EIGHT_clicked();

    void on_NINE_clicked();

    void on_LEFT_clicked();

    void on_RIGHT_clicked();

    void on_DIVIDE_clicked();

    void on_TIMES_clicked();

    void on_PLUS_clicked();

    void on_SUB_clicked();

    void on_ZEOR_clicked();

    void on_CLEAR_clicked();

    void on_DELETE_clicked();

    void on_EQUAL_clicked();

    void on_DIVIDE_2_clicked();

private:
    Ui::Widget *ui;
    QString expression;
};
#endif // WIDGET_H
